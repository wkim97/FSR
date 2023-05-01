python train.py --save_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0

python train.py --save_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0
python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0

python train.py --save_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 0
python test.py --load_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 0

python train.py --save_name svhn_resnet18 --dataset svhn --model resnet18 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 0

python train.py --save_name svhn_vgg16 --dataset svhn --model vgg16 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 0

python train.py --save_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 0