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

void test_init_value(){

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
    // 复杂情况的计算
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    Tensor rst = ts::div_pt(ts1, ts2);
    rst.backward();
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
};

int main(){
    Tensor ts1 = rand<double>({2, 3, 4});
    Tensor ts2 = rand<double>({2, 3, 4});
    ts1.set_require_grad(true);
    Tensor div_ = ts::div_pt(ts1, ts2);
//    cout << div_ << endl;
    div_.backward();

    cout << *ts2.get_node()->grad << endl;
    return 0;
}
