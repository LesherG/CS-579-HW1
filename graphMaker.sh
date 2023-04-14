

for x in "LeNet" "VGG16" "ResNet18"
do
    for y in "MNIST" "CIFAR"
    do
        "C:\Users\gavin\AppData\Local\Programs\Python\Python39\python.exe" ./plotting.py "$x" "$y"
    done
done