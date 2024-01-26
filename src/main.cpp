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


void test_autograd_initization(){
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

void test_grad_sub(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    ts1.set_require_grad(true);
    Tensor ts3 = ts1 - ts2;
    cout << ts1 << endl;
    cout << ts2 << endl;
    cout << ts3 << endl;
    ts3.backward();
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
}

void test_grad_mul(){
    Tensor ts1 = rand<double>({2, 3});
    Tensor ts2 = rand<double>({2, 3});
    ts1.set_require_grad(true);
    Tensor ts3 = ts::mul_pt(ts1, ts2);
    cout << ts1 << endl;
    cout << ts2 << endl;
    cout << ts3 << endl;
    ts3.backward();
    cout << *ts1.get_node()->grad << endl;
    cout << *ts2.get_node()->grad << endl;
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

int main(){
    test_211_indexing();
//2.menmory-sharing
//

//
//
//
////5.cat
//    int new_arr[3]{1,2,3};
//
//    Tensor t_cat2=Tensor(new_arr);
//    t1=Tensor(new_arr);
//
//    std::pair tensor_pair=std::make_pair(t1,t_cat2);
//
//
//    cout<<"tensor1:"<<t1<<endl;
//    cout<<"tensor2:"<<t_cat2<<endl;
////t_new=cat(tensor_pair,0);
////cout<<t_new<<endl;
////cout<<"cat_tensor:"<<cat(tensor_pair,1);
///*
//cat two tensor at dimension-0
//*/
//
//
////6.tile
//    vector<int> set;
//    t1=Tensor(given_arr);
//
//    set.push_back(1);
//    set.push_back(2);
//    cout<<"original tensor"<<t1<<endl;
//    cout<<"tile tensor:"<<t1.tile(set)<<endl;
///*
//the element appear once at dim 0,and twice at dim 1
//and the brodcast is also implement
//*/
//
//
////7.mutate
//    cout<<"original one"<<t1<<endl;
//    set.clear();
//    set.push_back(3);
//    set.push_back(2);
////cout<<t1.permute(set)<<endl;
//
//
//
////8.tranpose
//    t1=rand<int>(shape);
//    cout<<"initial_tensor"<<t1<<endl;
//    cout<<"transopose"<<t1.transpose(0,1)<<endl;
//
///*
//transpose at dim 0 as well as dim 1
//*/
//
//
////8.menmory_sharing
//    cout<<"data_sharing_same_menmory?"<<(&(t1.data_ptr()[0])==&(t1.transpose(0,1).data_ptr()[0]))<<endl;
//
//
//
//
////second-part-end
//
////thrid-part begin
//    int arr[2][3]{{1,2,3},{2,3,4}};
//    int arr2[2][3]{{0,0,1},{2,1,0}};
//
//    t1=Tensor(arr);
//    Tensor t2=Tensor(arr2);
//
//    cout<<"tensor 1"<<t1<<endl;
//    cout<<"tensor 2"<<t2<<endl;
//    cout<<"+:"<<t1+t2<<endl;
//    cout<<"add"<<add(t1,t2)<<endl;
//
///*
//overwritting operator +
//and the method add
//and static method add is aslo ok
//*/
//
//
////thrid-part end
//
//// Part4. Autograd
//
///*
// * Introduction.
// * 我们实现了基于计算图的 autograd 框架，所有的节点都可以通过 set_require_grad 的方法来生成背后的节点。
// * 这个节点主要拥有 value 和 grad 属性，分别对应其值、梯度值，这个计算是延迟计算的。
// * 他们将分别通过正向、反向的方式进行计算。
// */
//
//
//
////    test_grad_sub();
//
////    test_inv_pt();
//
//test_grad_mul();
    return 0;

}
