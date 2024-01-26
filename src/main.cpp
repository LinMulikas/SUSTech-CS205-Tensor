#include <iostream>
#include <vector>
#include "tensor.h"
#include "grad.h"


using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using std::cout, std::endl, std::vector;

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

void test_inv_pt(){
    Tensor ts1 = rand<double>({2, 3});
    cout << ts1 << endl;
    cout << ts::inv_pt(ts1) << endl;
}

int main(){


// Part4. Autograd

/*
 * Introduction.
 * 我们实现了基于计算图的 autograd 框架，所有的节点都可以通过 set_require_grad 的方法来生成背后的节点。
 * 这个节点主要拥有 value 和 grad 属性，分别对应其值、梯度值，这个计算是延迟计算的。
 * 他们将分别通过正向、反向的方式进行计算。
 */



//    test_grad_sub();

//    test_inv_pt();

}
