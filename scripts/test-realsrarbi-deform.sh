echo 'realsrarbi-x1.5' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-1dot5.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x2' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-2.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x2.5' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-2dot5.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x3' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-3.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x3.5' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-3dot5.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x4' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-4.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x1.7' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-1dot7.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x2.3' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-2dot3.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x2.7' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-2dot7.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x3.3' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-3dot3.yaml --model $1 --gpu $2 &&

echo 'realsrarbi-x3.7' &&
python test_real_deform.py --config ./configs/test/test-realsrarbi-3dot7.yaml --model $1 --gpu $2 &&


true
