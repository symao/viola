import os

cur_path = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.listdir(os.path.join(cur_path,'../src'))[0]
src_dir = os.path.join(cur_path, '../src/',sub_dir)
headers = sorted([f for f in os.listdir(src_dir) if f[:3] == 'vs_' and f[-2:] == '.h'])

def check_header(header, key):
    abs_header = os.path.join(src_dir,header)
    abs_cpp = abs_header.replace('.h','.cpp')
    for l in lines:
        if key in l:
            return True
    for l in lines:
        if '#include "vs_' in l:
            depend_header = l.split('"')[1]
            flag = check_header(depend_header,key)
            if flag:
                return True
    return False

def check_headers(key):
    find_list = []
    for h in headers:
        abs_header = os.path.join(src_dir,h)
        abs_cpp = abs_header.replace('.h','.cpp')
        lines = open(abs_header).readlines() + (open(abs_cpp).readlines() if os.path.exists(abs_cpp) else [])
        for l in lines:
            if key in l:
                find_list.append(h)
                break
    for _ in range(5):
        for h in headers:
            if h in find_list:
                continue
            abs_header = os.path.join(src_dir,h)
            abs_cpp = abs_header.replace('.h','.cpp')
            lines = open(abs_header).readlines() + (open(abs_cpp).readlines() if os.path.exists(abs_cpp) else [])
            for l in lines:
                if '#include "vs_' in l:
                    depend_header = l.split('"')[1]
                    if depend_header in find_list:
                        find_list.append(h)
                        break
    return find_list

with open('header_summary.md', 'w') as fp:
    fp.write('|Filename|Need Eigen|Need OpenCV|Details|\n'
             '|--|--|--|--|\n')
    cv_headers = check_headers("<opencv2/")
    eigen_headers = check_headers("<Eigen/")
    for h in headers:
        # read details
        detail_strs = [l for l in open(os.path.join(src_dir,h)).readlines() if '@details' in l]
        detail_str = detail_strs[0].split('@details')[1].strip() if len(detail_strs) > 0 else ''
        fp.write('|[%s](./src/%s/%s)|%s|%s|%s|\n'%(h,sub_dir,h,
                '√' if h in eigen_headers else '',
                '√' if h in cv_headers else '',
                detail_str))

        

