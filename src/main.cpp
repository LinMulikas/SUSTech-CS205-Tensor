#include <iostream>
#include "tensor.h"




int main() {
    double array[2][2][2] = {{{1, 2 }, {3, 4}},{{1, 2 }, {3, 4}} };
//    ts::Tensor tensor(array);
    int size[3] = {2, 2, 1};
    ts::Tensor tensor(array);
//    ts::Tensor tensor = ts::Tensor::rand<double>(size);
    ts::Tensor tensor1 = ts::rand<double>(size);
    std::cout << tensor << std::endl;
    std::cout << tensor.transpose(1, 2) << std::endl;

//    tensor(1, {2, 3}) = size;
    std::cout << tensor << std::endl;
    std::cout << ts::tile(tensor, size) << std::endl;

    return 0;
}
