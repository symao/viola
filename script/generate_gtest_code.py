import os

cur_path = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(cur_path, '../src/viola')
gtest_dir = os.path.join(cur_path, '../gtest')

headers = [f for f in os.listdir(src_dir) if f[:3] == 'vs_' and f[-2:] == '.h']
for f in headers:
    gtest_file = os.path.join(gtest_dir, 'test_'+f[3:-2]+'.cpp')
    if not os.path.exists(gtest_file):
      with open(gtest_file, 'w') as fp:
        fp.write('#include <gtest/gtest.h>\n')
        fp.write('#include <viola/viola.h>\n')
        fp.write('using namespace vs;\n')