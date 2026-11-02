"""Contract, negative, FileCheck, and idempotency tests for the real C++ pass."""
from pathlib import Path
import subprocess
import sys
import unittest

TOOL=sys.argv.pop(1)
ROOT=Path(__file__).parent
SOURCE=(ROOT/'contracts.mlir').read_text()


def invoke(text, lower=True):
    args=[TOOL,'--verify-each']
    if lower:args.append('--s9-lower-aten-mean')
    return subprocess.run(args,input=text,text=True,capture_output=True)


class Contracts(unittest.TestCase):
    def test_warp_contract(self):
        source='''module {
          llvm.func @mean(%p: i64) -> f32 {
            %v = "s9.aten_mean_warp"(%p) {rows = ROWS : i64,
              contract = "pytorch-2.8.0-cuda-bf16-square-mean-warp32-v1"} : (i64) -> f32
            llvm.return %v : f32
          }
        }'''
        for rows in (1,2,3,4,7,8,15,16,33,346,4096):
            result=invoke(source.replace('ROWS',str(rows)))
            self.assertEqual(result.returncode,0,result.stderr)
            ir=result.stdout
            self.assertNotIn('"s9.aten_mean_warp"',ir)
            self.assertNotIn('fastmath',ir)
            self.assertNotIn('vector<1536',ir)
            # 12 groups of four BF16 loads per physical lane, exactly 48
            # elements, regardless of the ATen logical block width.
            self.assertEqual(ir.count('llvm.load'),12)
            offsets=[ir.index(f'shfl.sync.down.b32 $0, $1, {i},') for i in (1,2,4,8,16)]
            self.assertEqual(offsets,sorted(offsets))
            self.assertIn('shfl.sync.idx.b32 $0, $1, 0,',ir)
            self.assertIn('has_side_effects',ir)
            self.assertEqual(invoke(ir).stdout,ir)
        for old,new in [('ROWS','0'),('ROWS','4097'),
                        ('warp32-v1','bad'),('contract =','extra = true, contract ='),
                        ('%p: i64','%p: i32')]:
            text=source.replace(old,new).replace('ROWS','1')
            self.assertNotEqual(invoke(text).returncode,0)

    def test_filecheck_and_barriers(self):
        result=invoke(SOURCE)
        self.assertEqual(result.returncode,0,result.stderr)
        checked=subprocess.run(['/usr/lib/llvm-19/bin/FileCheck',str(ROOT/'contracts.mlir'),
            '--implicit-check-not=s9.aten_mean','--implicit-check-not=fastmath'],
            input=result.stdout,text=True,capture_output=True)
        self.assertEqual(checked.returncode,0,checked.stderr)

    def test_no_pass(self):
        result=invoke(SOURCE,False)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('"s9.aten_mean"',result.stdout)

    def test_idempotency(self):
        once=invoke(SOURCE)
        twice=invoke(once.stdout)
        self.assertEqual(twice.returncode,0,twice.stderr)
        self.assertEqual(once.stdout,twice.stdout)

    def test_rows_dispatch_boundaries(self):
        for rows in (1,2,3,4,7,8,15,16,33,346,4096):
            with self.subTest(rows=rows):
                result=invoke(SOURCE.replace('rows = 346','rows = '+str(rows)))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertNotIn('"s9.aten_mean"',result.stdout)

    def test_reject_invalid_contracts(self):
        cases=[
            ('rows = 346','rows = 0','requires rows'),
            ('rows = 346','rows = -1','requires rows'),
            ('rows = 346','rows = 4097','requires rows'),
            ('rows = 346 : i64','rows = 346 : i32','requires rows'),
            ('rows = 346 : i64, ','','requires rows'),
            ('pytorch-2.8.0-cuda-f32-mean-v1','unverified','explicit PyTorch'),
            ('vector<1536xf32>','vector<1535xf32>','requires vector'),
            ('vector<1536xf32>','vector<1536xf16>','requires vector'),
            ('contract =','extra = true, contract =','unsupported attribute'),
        ]
        # Only the contract function, to avoid unrelated ordinary-reduction errors.
        function=SOURCE.split('// CHECK-LABEL: func.func @rounding_barrier')[0]
        for old,new,message in cases:
            with self.subTest(old=old,new=new):
                result=invoke(function.replace(old,new))
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
