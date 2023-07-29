sudo apt install build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev libgtk-3-dev libvtk9-dev libavcodec-dev libavformat-dev libswscale-dev

sudo apt install libatlas-base-dev libeigen3-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev

mkkdir build; cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../extra_modules -DWITH_TBB=OFF -DOPENCV_ENABLE_NONFREE=ON -DWITH_GTK=ON -DWITH_OPENGL=ON -DBUILD_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_TEST=OFF -DINSTALL_C_EXAMPLES=OFF

make -j4