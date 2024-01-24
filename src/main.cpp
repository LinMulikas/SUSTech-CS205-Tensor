#include <iostream>
#include <vector>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using std::cout, std::endl, std::vector;


int main(){
    Tensor ts1 = rand<double>({2, 3, 4});
    cout << ts1 << endl;
    cout << ts1.mean(2, false) << endl;
    return 0;
}
