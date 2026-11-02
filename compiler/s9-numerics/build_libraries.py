"""Lower an explicit ATen arithmetic contract, produce CPU and GPU libraries.

The input is an extracted/normalized RMSNorm mean region, not an unmodified
Triton module. The bitcode is subsequently linked into an actual Triton kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(args, output=None):
    result = subprocess.run([str(x) for x in args],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'command failed: {args}\n{result.stderr}')
    if output: Path(output).write_text(result.stdout)
    return result.stdout


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tool',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--rows',type=int,nargs='+',default=[1,346])
    ap.add_argument('--llvm',default='/usr/lib/llvm-19/bin')
    ap.add_argument('--parallel',action='store_true',help='one physical warp per row; retain scalar CPU oracle')
    args=ap.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    llvm=Path(args.llvm)
    manifest={}
    for rows in args.rows:
        stem=f'mean_1536_r{rows}'
        before=root/f'{stem}.before.mlir'
        after=root/f'{stem}.after.mlir'
        lowered=root/f'{stem}.llvm.mlir'
        ll=root/f'{stem}.ll'
        bc=root/f'{stem}.bc'
        so=root/f'{stem}.so'
        before.write_text('''module {
  llvm.func @s9_aten_mean(%address: i64) -> f32 {
    %p = llvm.inttoptr %address : i64 to !llvm.ptr<1>
    %x = llvm.load %p {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1536xbf16>
    %f = arith.extf %x : vector<1536xbf16> to vector<1536xf32>
    %sq = arith.mulf %f, %f : vector<1536xf32>
    %v = "s9.aten_mean"(%sq) {rows = ROWS : i64,
      contract = "pytorch-2.8.0-cuda-f32-mean-v1"} : (vector<1536xf32>) -> f32
    llvm.return %v : f32
  }
}
'''.replace('ROWS',str(rows)))
        run([args.tool,before,'--s9-lower-aten-mean','--verify-each'],after)
        run([llvm/'mlir-opt',after,'--convert-vector-to-llvm',
             '--convert-arith-to-llvm','--reconcile-unrealized-casts'],lowered)
        run([llvm/'mlir-translate',lowered,'--mlir-to-llvmir'],ll)
        # Avoid handing a 1536-lane vector load to NVPTX legalization. This
        # mechanical LLVM scalarization preserves the explicit FP32 add tree.
        rawbc=root/f'{stem}.raw.bc'
        run([llvm/'llvm-as',ll,'-o',rawbc])
        run([llvm/'opt','-passes=scalarizer,instcombine,simplifycfg',
             '-scalarize-load-store',rawbc,'-o',bc])
        run([llvm/'clang','-shared','-fPIC','-O2','-Wno-override-module',ll,'-o',so])
        if args.parallel:
            stem += '_warp'
            before,after,lowered,ll,bc=[root/f'{stem}.{suffix}' for suffix in
                                      ('before.mlir','after.mlir','llvm.mlir','ll','bc')]
            before.write_text('''module {
  llvm.func @s9_aten_mean(%address: i64) -> f32 attributes {passthrough = ["convergent", "noduplicate"]} {
    %v = "s9.aten_mean_warp"(%address) {rows = ROWS : i64,
      contract = "pytorch-2.8.0-cuda-bf16-square-mean-warp32-v1"} : (i64) -> f32
    llvm.return %v : f32
  }
}
'''.replace('ROWS',str(rows)))
            run([args.tool,before,'--s9-lower-aten-mean','--verify-each'],after)
            run([llvm/'mlir-opt',after,'--convert-vector-to-llvm',
                 '--convert-arith-to-llvm','--reconcile-unrealized-casts'],lowered)
            run([llvm/'mlir-translate',lowered,'--mlir-to-llvmir'],ll)
            run([llvm/'llvm-as',ll,'-o',bc])
        manifest[f'{rows}x1536']=dict(rows=rows,width=1536,bitcode=bc.name,
            cpu_library=so.name,sha256=hashlib.sha256(bc.read_bytes()).hexdigest(),
            implementation='warp32' if args.parallel else 'scalar',
            num_warps=1 if args.parallel else 4)
    (root/'libraries.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
