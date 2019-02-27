export LIBSHARP=~/miniconda
git clone https://github.com/Libsharp/libsharp --branch v1.0.0 --single-branch --depth 1 \
    && cd libsharp \
    && autoreconf \
    && CC="mpicc" \
    ./configure --enable-mpi --enable-pic \
    && make -j4 \
    && cp -a auto/* $LIBSHARP \
    && cd python \
    && CC="mpicc -g" LDSHARED="mpicc -g -shared" \
    pip install .
