#include <iostream>
#include "tensor.h"
using ts::Tensor;

using std::cout, std::endl;



int main(){
    // double array[2][2][2] = {
    //     {{1, 2}, {3, 4}},
    //     {{5, 6}, {7, 8}}};

    // double array1[3][5] = {{0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9},
    //                        {4.9, 5.2, 6.3, 7.4, 8.5}};

    // ts::Tensor ts1{array};
    // std::cout << ts1 << std::endl;
    // std::cout << ts1(1)(1)(1) << std::endl;
    // ts1(1)(1).showShape();

    // Test autograd.

    // Simple operators test.
    // - Add

    // zeros_like
    int shape1[1]{6};

    auto ts1 = ts::zeros<float>(shape1);
    cout << ts1 << endl;

    auto ts2 = ts::zeros_like(ts1);
    cout << ts2 << endl;
    // - Simple case of add, sub.

    return 0;
}
