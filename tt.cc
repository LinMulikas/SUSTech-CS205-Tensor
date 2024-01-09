#include <variant>
#include <iostream>

using namespace std;

using VD = variant<int, float>;

int main(){
    VD vd1{1};
    VD vd2{(float)1.1};



    return 0;
}