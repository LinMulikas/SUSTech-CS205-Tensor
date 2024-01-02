#include <iostream>
#include "tensor.h"




int main() {
    double array[2][1][2] = {{{1, 3}},{{5 , 7}} };
    double array1[3][5] = {{0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9},
                           {4.9, 5.2, 6.3, 7.4, 8.5}};

//    ts::Tensor tensor(array);
    int size[2] = {1, 15};
    ts::Tensor tensor(array1);
//    ts::Tensor tensor = ts::Tensor::rand<double>(size);
//    ts::Tensor tensor1 = ts::rand<double>(size);
    std::cout << tensor << std::endl;

    std::cout << ts::view(tensor, size) << std::endl;
//    std::cout << tensor.view(size) << std::endl;
//    std::cout << tensor.transpose(1, 2) << std::endl;

//    tensor(1, {2, 3}) = size;
//    std::cout << tensor << std::endl;
//    std::cout << ts::tile(tensor, size) << std::endl;

    return 0;
}
