#include <iostream>
#include <vector>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using std::cout, std::endl, std::vector;

/*
 * Init
 */

void test_11_init_copy(){
    // The raw data.
    int data[2][3] = {{1, 2, 3},
                      {4, 5, 6}};
    cout << "Init with raw data: " << endl;
    Tensor tensor{data};
    cout << tensor << endl;
}

void test_12_init_random(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<int>({3, 4});
    cout << ts1 << endl;
    cout << ts2 << endl;
}

void test_131_init_zero(){
    int shape[3]{2, 3, 4};
    Tensor ts1 = ts::zeros<double>(shape, 3);
    cout << ts1 << endl;
}

void test_132_init_ones(){
    int shape[3]{2, 3, 4};
    Tensor ts1 = ts::ones<double>(shape, 3);
    cout << ts1 << endl;
}

void test_25_view(){
    int shape[3]{2, 3, 4};
    Tensor t1 = rand<int>(shape);
    cout << "initial_tensor" << t1 << endl;
    cout << "new view:" << t1.view({4, 3, 2});
    cout << "data_sharing_same_menmory?" << endl;
    for(int i = 0; i < t1.get_total_size(); i++){
        cout << (&(t1.data_ptr()[i]) == &(t1.view({4, 3, 2}).data_ptr()[i])) << " ";
        cout << endl;
    }
/*
given tensor can view at the differnt shape
and the menmory will not change at all
*/

}

void test_211_indexing(){
    /*
     * Indexing 通过重载运算符 () 实现，每次传入一个参数表示寻找当前第一维的某个 index.
     */

    Tensor t1 = rand<double>({2, 5});
    cout << "t1:\n" << t1 << endl;
    std::cout << "indexing at [1]: \n" << t1(1) << std::endl;
    std::cout << "indexing at [1, 2]:\n" << t1(1)(2) << std::endl;

    /*
     * The data's adress are the same as before.
    */
    Tensor t_new = t1(1);
    // 新 tensor 是原来 tensor 的第二行，地址从原数据的第 6 个开始。
    std::cout << "Share_same_data_memory?\n" << ((&(t1.data_ptr()[5])) == (&(t_new.data_ptr()[0]))) << std::endl;

}

void test_212_slicing(){
    Tensor t1 = rand<double>({2, 3});
    std::pair one_pair = std::make_pair(2, 4);
    std::cout << "slice:" << t1(1, one_pair) << std::endl;
    /*
    slice is the same as index
    so as its multiply-form
    */


    //4.menmory_sharing
    Tensor t_new_2 = t1(1, one_pair);
    std::cout << "share_same_data_memory?" << ((&(t1.data_ptr()[4])) == (&(t_new_2.data_ptr()[0]))) << std::endl;
    std::cout << "share_same_data_memory?" << ((&(t1.data_ptr()[5])) == (&(t_new_2.data_ptr()[1]))) << std::endl;
    /*
    they are the same address
    */
}

void test_221_cat(){
    int new_arr[3]{1, 2, 3};

    Tensor t_cat2 = Tensor(new_arr);
    Tensor t1 = Tensor(new_arr);
    std::pair tensor_pair = std::make_pair(t1, t_cat2);
    cout << "tensor1:" << t1 << endl;
    cout << "tensor2:" << t_cat2 << endl;
    Tensor t_new = ts::cat(tensor_pair, 0);
    cout << "cat_tensor" << endl << t_new << endl;

    /*
    cat two tensor at dimension-0
    */
}

void test_222_tile(){

    vector<int> set;
    int given_arr[2][3]{{1, 2, 3},
                        {4, 5, 6}};
    Tensor t1 = Tensor(given_arr);

    set.push_back(1);
    set.push_back(2);
    cout << "original tensor" << t1 << endl;
    cout << "tile tensor:" << t1.tile(set) << endl;
/*
the element appear once at dim 0,and twice at dim 1
and the brodcast is also implement
*/
}

void test_241_transpose(){
    std::cout << "test transpose" << std::endl;
    double array[2][3][4] = {{{1, 3, 5, 7},  {2, 4, 6, 8},  {3, 5, 7,  9}},
                             {{4, 6, 8, 10}, {5, 7, 9, 11}, {6, 8, 10, 12}}};
    Tensor ts1(array);

    Tensor ts2 = ts1.transpose(0, 1);
    if(ts2.get_shape()[0] == 3 && ts2.get_shape()[1] == 2 && ts2.get_shape()[2] == 4){
        std::cout << "shape correct" << std::endl;
        bool flag = false;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 4; k++){
                    if(*(ts2(j)(i)(k).data_ptr()) != *(ts1(i)(j)(k).data_ptr())){
                        flag = true;
                        break;
                    }
                }
            }
        }
        if(flag){
            std::cout << "transpose incorrect" << std::endl;
        }else{
            std::cout << "transpose correct" << std::endl;
        }
    }else{
        std::cout << "transpose incorrect" << std::endl;
    }
}

void test_242_permute(){
    std::cout << "test_permute" << std::endl;
    double array[2][3][4] = {{{1, 3, 5, 7},  {2, 4, 6, 8},  {3, 5, 7,  9}},
                             {{4, 6, 8, 10}, {5, 7, 9, 11}, {6, 8, 10, 12}}};
    Tensor ts1(array);

    int size[3] = {2, 0, 1};
    Tensor ts2 = ts1.permute(size);
    if(ts2.get_shape()[0] == 4 && ts2.get_shape()[1] == 2 && ts2.get_shape()[2] == 3){
        std::cout << "shape correct" << std::endl;
        bool flag = false;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 4; k++){
                    if(*(ts2(k)(i)(j).data_ptr()) != *(ts1(i)(j)(k).data_ptr())){
                        std::cout << i << " " << j << " " << k << std::endl;
//                        std::visit(VariantPrinter{}, *(ts2(k)(i)(j).data_ptr()));
//                        std::visit(VariantPrinter{}, *(ts1(i)(j)(k).data_ptr()));
                        flag = true;
                        break;
                    }
                }
            }
        }
        if(flag){
            std::cout << "permute incorrect" << std::endl;
        }else{
            std::cout << "permute correct" << std::endl;
        }
    }else{
        std::cout << "shape incorrect" << std::endl;
    }
}

/*

/*
all thie method have to check the shape
and the point_wise-method must have same shape
while mul_dot must have shape that can mtach
*/
void test_311_add(){
    /*
       overwritting operator +
        and the method add
        and static method add is aslo ok
    */
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 + t2" << endl << t1 + t2 << endl;

}

void test_312_sub(){
    /*
  overwritting operator -
and the method sub
and static method add is aslo ok
    */
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 - t2" << endl << t1 - t2 << endl;

}

void test_313_mul_pt(){
    /*
   overwritting operator *
and the method multiply
and static method add is aslo ok
    */
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 * t2" << endl << t1.mul_pt(t2) << endl;

}

void test_316_dot(){
    /*
     * only last two dimension match can be valid
    */
    int arr[2]{2, 3};
    int arr1[2]{3, 2};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr1);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 · t2" << endl << t1.mul_dot(t2) << endl;
}

void test_314_div(){
    /*
    only "div_pt" can be effect
    */
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 / t2" << endl << t1.div_pt(t2) << endl;

}

void test_315_log(){
    /*
    only "div_pt" can be effect
    */
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor log_t1 = ts::apply(t1, log);
    cout << log_t1 << endl;

}


/*
compare with each other point-wise
*/

void test_334_ge(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 >= t2?" << endl << t1.ge(t2) << endl;

}

void test_336_le(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 <= t2?" << endl << t1.le(t2) << endl;

}

void test_331_eq(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 == t2?" << endl << t1.eq(t2) << endl;

}

void test_332_ne(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 == t2?" << endl << t1.ne(t2) << endl;

}

void test_333_gt(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 > t2?" << endl << t1.gt(t2) << endl;

}

void test_335_lt(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    Tensor t2 = rand<int>(arr);
    cout << "tensor1:" << endl << t1 << endl;
    cout << "tensor2:" << endl << t2 << endl;
    cout << "t1 < t2?" << endl << t1.lt(t2) << endl;

}


/*
these method all can choose keep-dim or not
and here we use false means do not keep dimension
*/

void test_322_mean(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    cout << "tensor" << endl << t1 << endl;
    cout << "mean:" << endl << t1.mean(0, false) << endl;
}

void test_323_max(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    cout << "tensor" << endl << t1 << endl;
    cout << "max:" << endl << t1.max(0, false) << endl;
}

void test_324_min(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    cout << "tensor" << endl << t1 << endl;
    cout << "min:" << endl << t1.min(0, false) << endl;
}

void test_331_sum(){
    int arr[2]{2, 3};
    Tensor t1 = rand<int>(arr);
    cout << "tensor" << endl << t1 << endl;
    cout << "sum:" << endl << t1.sum(0, false) << endl;
}


// test 4.3 Aurograd

void test_431_initization(){
    // 4.1 隐式生成计算节点
    // 4.1.1 不生成梯度
    Tensor grad_ts1 = rand<double>({2, 3});
    Tensor grad_ts2 = rand<double>({2, 3});
    cout << grad_ts1 + grad_ts2 << endl;
    cout << grad_ts1.get_node() << endl; // nullptr
    // 创建 node
    grad_ts1.set_require_grad(true);
    // node 也可以反向访问其值
    cout << *grad_ts1.get_node()->value << endl;
    // 当一个 Tensor 生成了节点之后，使用该节点的计算会默认使得其他参与计算的节点也生成 node
    Tensor grad_added = grad_ts1 + grad_ts2;
    cout << grad_added.get_node() << endl; // 由于 ts1，进而产生了 grad
    /*
        * 显示也可以构造节点
        */
    grad::AddNode added_node{grad_ts1.get_node(), grad_ts2.get_node()};
    added_node.forward();
    cout << *added_node.value << endl;
}

// 4.3.2 各种节点的测试
void test_4321_grad_add(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    // 指定任意节点需要梯度，那么结果都会需要梯度。
    ts1.set_require_grad(true);
    Tensor ts3 = ts1 + ts2;
    cout << "ts1, ts2:\n" << endl;
    cout << ts1 << endl;
    cout << ts2 << endl;
    cout << "ts3 = ts1 + ts2:\n" << endl;
    cout << ts3 << endl;
    ts3.backward();
    cout << "grad of ts1, ts2:\n" << endl;
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
    // add 的情况下，每一个的梯度都是 ones
}

void test_4322_grad_sub(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    ts1.set_require_grad(true);
    Tensor ts3 = ts1 - ts2;
    cout << ts1 << endl;
    cout << ts2 << endl;
    cout << ts3 << endl;
    ts3.backward();
    // sub 情况下，ts1 梯度是 ones, ts2 梯度是 -ones.
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
}

void test_4323_grad_mul(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    ts1.set_require_grad(true);
    Tensor ts3 = ts::mul_pt(ts1, ts2);
    cout << "ts1, ts2:" << endl;
    cout << ts1 << endl;
    cout << ts2 << endl;
    cout << "ts3 = ts1 * ts2:" << endl;
    cout << ts3 << endl;
    ts3.backward();
    cout << "grad of ts1, ts2: " << endl;
    // 分别是对方的值
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
}

void test_4324_div(){
    // div
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    Tensor rst = ts::div_pt(ts1, ts2);
    rst.backward();
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
};

int main(){
//    test_11_init_copy();
//    test_12_init_random();
//    test_131_init_zero();
//    test_132_init_ones();
//    test_211_indexing();
//    test_212_slicing();
//    test_221_cat();
//    test_222_tile();
//    test_241_transpose();
//    test_242_permute();
//    test_25_view();
//    test_311_add();
//    test_312_sub();
//    test_313_mul_pt();
//    test_314_div();
//    test_315_log();
//    test_316_dot();
//
//
//    test_331_eq();
//    test_332_ne();
//    test_333_gt();
//    test_334_ge();
//    test_335_lt();
//    test_336_le();
//
//    test_331_sum();
//    test_322_mean();
//    test_323_max();
//    test_324_min();

//    test_431_initization();
//    test_4321_grad_add();
//    test_4322_grad_sub();
//    test_4324_div();

    return 0;
}
