for i in 256 512 1024
do
    echo -n "cufft,$i,"
    ./cufft_f32_speed -b 1048576 -n $i
    echo -n "tcfft,$i,"
    ./tcfft_f32_speed -b 1048576 -n $i
    echo -n "mpmxufft,$i,"
    ./mpmxufft_f32_speed -b 1048576 -n $i
done

for i in 131072 262144 524288
do
    echo -n "cufft,$i,"
    ./cufft_f32_speed -b 2048 -n $i
    echo -n "tcfft,$i,"
    ./tcfft_f32_speed -b 2048 -n $i
    echo -n "mpmxufft,$i,"
    ./mpmxufft_f32_speed -b 2048 -n $i
done

for i in 16777216 33554432 67108864 134217728
do
    echo -n "cufft,$i,"
    ./cufft_f32_speed -b 8 -n $i
    echo -n "tcfft,$i,"
    ./tcfft_f32_speed -b 8 -n $i
    echo -n "mpmxufft,$i,"
    ./mpmxufft_f32_speed -b 8 -n $i
done
