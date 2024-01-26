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

<<<<<<< HEAD
  //tensor的init()
//1.given array
int given_arr[2][3]{{1,2,3},{4,5,6}};
Tensor t1=Tensor(given_arr);
cout<<t1<<endl;
/*
the given data as a argument so we'll just make 
a new tensor and copy data on it.
*/
//2.given-shape and randomly
int shape[]{2,3,4};
t1=rand<double>(shape);
/*
shape 可自定，rand可以传入任意的variantdata
double，bool，int，float类型
*/
cout<<t1<<endl;
//3.initial with zeros
int shape1[]{3,2};
t1=zeros<bool>(shape1);
/*
得到全0的tensor
shape 可自定，rand可以传入任意的variantdata
double，bool，int，float类型
*/
cout<<t1<<endl;

//4.initial with ones
int shape2[]{2,2};
Tensor t=ts::ones<int>(shape2);
cout<<"vec for all ones"<<t<<endl;
/*
得到的是全1的tensor
shape可自定
*/


//5.initial with given value
int shape4[2]{2,2};
int given_value[2][2]{{1,2},{3,4}};


cout<<"initial_with_given_arr"<<t<<endl;

/*
make a new tensor with given shape and data type
*/

//6.pattern




  //end

//second-part-begin

//1.indexing
t1=Tensor(given_arr);
std::cout<<"index"<<t1(1)(2)<<std::endl;
/*
the index of the t1.
that's the index of 1,2
index start from zero

multi-index is aslo ok,
just use it like t(1)(2)(3)
*/
//2.menmory-sharing

Tensor t_new=t1(1)(2);
std::cout<<"share_same_data_memory?"<<((&(t1.data_ptr()[5]))==(&(t_new.data_ptr()[0])))<<std::endl;
/*
the data's adress are the 
same as before
*/

//3.
std::pair one_pair=std::make_pair(2,4);
std::cout<<"slice:"<<t1(1,one_pair)<<std::endl;
/*
slice is the same as index
so as its multiply-form
*/


//4.menmory_sharing
Tensor t_new_2=t1(1,one_pair);
std::cout<<"share_same_data_memory?"<<((&(t1.data_ptr()[4]))==(&(t_new_2.data_ptr()[0])))<<std::endl;
std::cout<<"share_same_data_memory?"<<((&(t1.data_ptr()[5]))==(&(t_new_2.data_ptr()[1])))<<std::endl;
/*
they are the same address
*/


//5.cat
int new_arr[3]{1,2,3};

Tensor t_cat2=Tensor(new_arr);
t1=Tensor(new_arr);

std::pair tensor_pair=std::make_pair(t1,t_cat2);


cout<<"tensor1:"<<t1<<endl;
cout<<"tensor2:"<<t_cat2<<endl;
//t_new=cat(tensor_pair,0);
//cout<<t_new<<endl;
//cout<<"cat_tensor:"<<cat(tensor_pair,1);
/*
cat two tensor at dimension-0
*/


//6.tile
vector<int> set;
t1=Tensor(given_arr);

set.push_back(1);
set.push_back(2);
cout<<"original tensor"<<t1<<endl;
cout<<"tile tensor:"<<t1.tile(set)<<endl;
/*
the element appear once at dim 0,and twice at dim 1
and the brodcast is also implement
*/


//7.mutate
cout<<"original one"<<t1<<endl;
set.clear();
set.push_back(3);
set.push_back(2);
//cout<<t1.permute(set)<<endl;



//8.tranpose
t1=rand<int>(shape);
cout<<"initial_tensor"<<t1<<endl;
cout<<"transopose"<<t1.transpose(0,1)<<endl;

//9.view
cout<<"initial_tensor"<<endl<<t1<<endl;

=======

// Part4. Autograd
>>>>>>> 370d84da6021b6849c3e9b587bef5c634a47235d

/*
 * Introduction.
 * 我们实现了基于计算图的 autograd 框架，所有的节点都可以通过 set_require_grad 的方法来生成背后的节点。
 * 这个节点主要拥有 value 和 grad 属性，分别对应其值、梯度值，这个计算是延迟计算的。
 * 他们将分别通过正向、反向的方式进行计算。
 */



//    test_grad_sub();

<<<<<<< HEAD
//second-part-end
   
//thrid-part begin
int arr[2]{2,3};
int arr2[2][3]{{0,0,1},{2,1,0}};
//1.add
t1=ts::ones<int>(arr);
Tensor t2=ts::ones<int>(arr);

cout<<"tensor 1"<<t1<<endl;
cout<<"tensor 2"<<t2<<endl;

cout<<"+:"<<t1+t2<<endl;
cout<<"add"<<add(t1,t2)<<endl;

/*
overwritting operator + 
and the method add
and static method add is aslo ok
*/
//2.sub
cout<<"-:"<<t1-t2<<endl;
cout<<"sub"<<endl<<sub(t1,t2)<<endl;

//3.div
cout<<"div"<<endl<<t1.div_pt(t2)<<endl;


//4.mul_pt
cout<<"mul_pt"<<endl<<t1.mul_pt(t2)<<endl;

//5.mul_dot
t1=rand<int>(arr);
int arr_new[2]{3,1};
t2=rand<int>(arr_new);
cout<<"mul_dot"<<endl<<t1.mul_dot(t2)<<endl;

/*
all thie method have to check the shape
and the point_wise-method must have same shape
while mul_dot must have shape that can mtach
*/


//1.ge
t1=rand<int>(arr);
t2=rand<int>(arr);
cout<<"tensor1:"<<endl<<t1<<endl;
cout<<"tensor2:"<<endl<<t2<<endl;
cout<<"tensor>=tensor2:"<<endl<<t1.ge(t2)<<endl;

//2.le
cout<<"tensor<=tensor2:"<<endl<<t1.le(t2)<<endl;

//3.eq
cout<<"tensor==tensor2:"<<endl<<t1.eq(t2)<<endl;

//4.gt
cout<<"tensor>tensor2:"<<endl<<t1.gt(t2)<<endl;

//5.lt
cout<<"tensor>=tensor2:"<<endl<<t1.lt(t2)<<endl;
/*
compare with each other point-wise
*/


//max
cout<<"tensor:"<<endl<<t1<<endl;
cout<<"max:"<<endl<<max(t1,0,false)<<endl;

//min
cout<<"min:"<<endl<<min(t1,0,false)<<endl;

//mean
cout<<"mean:"<<endl<<t1.mean(0,false)<<endl;

//sum
cout<<"sum:"<<endl<<t1.sum(0,false)<<endl;

/*
these method all can choose keep-dim or not
and here we use false means do not keep dimension
*/


//thrid-part end
=======
//    test_inv_pt();

>>>>>>> 370d84da6021b6849c3e9b587bef5c634a47235d
}
