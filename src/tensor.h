#pragma once

#include <variant>
#include <iostream>
#include <type_traits>
#include <iomanip>
#include <vector>
#include <memory>
#include <random>
#include <typeinfo>
#include <initializer_list>
#include "grad.h"

namespace ts{
using VariantData = std::variant<bool, int, float, double>;
using std::cout, std::endl;
using std::vector;
using std::shared_ptr, std::make_shared;

template<typename T>
using List = std::initializer_list<T>;

double promote(VariantData &vd);

void assign(VariantData &vd, double val, int type_id);

double inv(VariantData &vd) throw();

template<class T>
static int dtype_id_from(){
    if(typeid(T) == typeid(bool)) return 0;
    if(typeid(T) == typeid(int)) return 1;
    if(typeid(T) == typeid(float)) return 2;
    if(typeid(T) == typeid(double)) return 3;
    return 1;
}

static bool global_require_grad = false;

class Tensor{
private:
    std::shared_ptr<VariantData[]> data_shared;
    VariantData *data;

    int dimension;
    std::shared_ptr<int[]> shape;

    int total_size;
    int dtype_id;

    // Autograd
    bool _require_grad = false;
    shared_ptr<ts::Tensor> grad;
    shared_ptr<grad::Node> node;

    template<typename ArrayType, size_t N>
    void copyData(ArrayType(&arr)[N], VariantData *&dest, VariantData *destEnd){
        for(size_t i = 0; i < N; ++i){
            if constexpr(std::is_array<ArrayType>::value){
                copyData(arr[i], dest, destEnd);
            }else{
                *dest = arr[i];
                if(dest < destEnd)
                    dest++;
            }
        }
    }

    template<typename ArrayType, size_t N>
    void getShape(ArrayType(&arr)[N], int dim){
        shape.get()[dim] = N;
        if constexpr(std::is_array<ArrayType>::value)
            getShape(arr[0], dim + 1);
    }


    void print(std::ostream &os, int index, int dim) const;

    void test_dim_fn(Tensor(*fn)(Tensor &, int));

public:

    bool get_require_grad() const{
        return _require_grad;
    }

    void set_require_grad(bool flag);

    vector<int> get_shape_vec();

    void backward(){
        node->backward();
    }

    shared_ptr<grad::Node> &get_node();

    Tensor add(Tensor &t2);

    Tensor add(VariantData &vd);

    Tensor sub(Tensor &t2);

    Tensor sub(VariantData vd);

    Tensor mul_pt(Tensor &t2);

    Tensor mul_pt(VariantData vd);

    Tensor mul_dot(Tensor &t2) throw();

    friend Tensor mul_dot_2d(Tensor &t1, Tensor &t2);

    Tensor div_pt(Tensor &t2);

    Tensor div_pt(VariantData vd);

    //comparator
    Tensor eq(Tensor &t2) throw();

    Tensor ne(Tensor &t2) throw();

    Tensor ge(Tensor &t2) throw();

    Tensor gt(Tensor &t2) throw();

    Tensor le(Tensor &t2) throw();

    Tensor lt(Tensor &t2) throw();
    //comparator

    //dim-operator
    Tensor expansion_1d();

    Tensor mean(int, bool);

    Tensor sum(int dim, bool keepdim);

    Tensor min(int dim, bool);

    Tensor max(int dim, bool);


    static VariantData copy_tile(VariantData *src, Tensor *dst, int idx, int *src_shape, int dim);

    // init a tensor with exact const shape arr.
    template<size_t N>
    static Tensor init_with_shape(int(&size)[N]){
        Tensor t = Tensor();
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();
        // default type float.
        t.dtype_id = 2;
        return t;
    }

    // init a tensor with exact shape arr and dim N.
    static Tensor init_with_shape(const int size[], int N){
        Tensor t = Tensor();
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();

        // default type float.
        t.dtype_id = 2;
        return t;
    }

    // init a dtype T tensor with exact shape arr and dim N.
    // Have impled the autograd.
    template<typename T>
    static Tensor init_with_shape(const int size[], int N){
        Tensor t = Tensor(dtype_id_from<T>());

        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();

        t.dtype_id = dtype_id_from<T>();
        return t;
    }

    int *size();

    std::string type_name();

    int get_dtype_id() const{
        return dtype_id;
    }

    VariantData *data_ptr() const;

    int get_total_size() const;

    int *get_shape() const;

    int get_dimension() const;

    void showShape(){
        for(int i = 0; i < dimension; i++){
            std::cout << shape[i] << " ";
        }
        std::cout << std::endl;
    }


    // Constructors
    // - Init tensor with exact data.
    template<typename T, size_t N>
    explicit Tensor(T(&arr)[N]){
        // dimension of arr, e.g. double[2][1] dimension = 2
        dimension = std::rank<T>::value + 1;
        shape.reset(new int[dimension]);
        using BaseType = typename std::remove_all_extents<T>::type();
        // type_name = typeid(BaseType).name();
        bool flag = (typeid(T) == typeid(int[]));
//        cout <<  << endl;
        dtype_id = dtype_id_from<T>();

        getShape(arr, 0);

        // copy array to data
        total_size = 1;
        for(int i = 0; i < dimension; i++){
            total_size *= shape[i];
        }
        data_shared.reset(new VariantData[total_size]);
        data = data_shared.get();
        VariantData *pointer = data;
        copyData(arr, pointer, data_shared.get() + total_size);
    }

    template<typename T, size_t M, size_t  N>
    explicit Tensor(T arr[M][N], int dim){
        // dimension of arr, e.g. double[2][1] dimension = 2
        dimension = std::rank<T>::value + 1;
        shape.reset(new int[dimension]);
        using BaseType = typename std::remove_all_extents<T>::type();
        // type_name = typeid(BaseType).name();
        bool flag = (typeid(T) == typeid(int));
//        cout <<  << endl;
        dtype_id = dtype_id_from<T>();

        getShape(arr, 0);

        // copy array to data
        total_size = 1;
        for(int i = 0; i < dimension; i++){
            total_size *= shape[i];
        }
        data_shared.reset(new VariantData[total_size]);
        data = data_shared.get();
        VariantData *pointer = data;
        copyData(arr, pointer, data_shared.get() + total_size);
    }

    Tensor(int type_id);

    Tensor();

    int cal_stride(int dim, int *shape);

    friend std::ostream &operator<<(std::ostream &os, Tensor const &t);


    Tensor operator()(int idx) const;

    Tensor operator()(int idx, std::pair<int, int> range) const;

    void operator=(const VariantData &value);

    template<size_t N>
    void operator=(const int(&arr)[N]){
        copyData(arr, data, data + total_size);
    }

    Tensor transpose(int dim1, int dim2);

    Tensor permute(int dim[]);

    Tensor permute(vector<int> dimens);

    template<size_t N>
    Tensor view(int(&shape)[N]) throw(){
        size_t cnt_data = 1;
        for(int i = 0; i < N; i++){
            cnt_data *= shape[i];
        }
        if(cnt_data != total_size)
            throw std::invalid_argument("Invalid arg: The new shape should has same count of data.");
        Tensor t = Tensor();
        t.data = data;
        t.data_shared = data_shared;
        t.total_size = total_size;
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = shape[i];
        }
        return t;
    }

    Tensor view(vector<int> &&vec){
        int arr[vec.size()];
        for(int i = 0; i < vec.size(); i++){
            arr[i] = vec[i];
        }
        return view(arr, vec.size());
    }

    Tensor view(const int *shape, int dim){
        Tensor t = Tensor();
        t.data = data;
        t.data_shared = data_shared;
        t.total_size = total_size;
        t.dimension = dim;
        t.shape.reset(new int[t.dimension]);
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = shape[i];
        }
        return t;
    }

    template<typename T, size_t N>
    static Tensor eye(int(&size)[N]){
        Tensor t = init_with_shape(size);
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();
        for(int i = 0; i < t.total_size; i++){
            t.data[i] = (T) 0;
        }
        int min = t.shape[0] < t.shape[1] ? t.shape[0] : t.shape[1];
        for(int i = 0; i < min; i++){
            t.data[i * t.shape[1] + i] = (T) 1;
        }
        return t;
    }

    template<typename T>
    static Tensor eye(int *size, const int dim){
        Tensor t = init_with_shape<T>(size, dim);
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();
        for(int i = 0; i < t.total_size; i++){
            t.data[i] = (T) 0;
        }
        int min = t.shape[0] < t.shape[1] ? t.shape[0] : t.shape[1];
        for(int i = 0; i < min; i++){
            t.data[i * t.shape[1] + i] = (T) 1;
        }
        return t;
    }


    void test_sum();

    void test_mean();

    void test_max();

    void test_min();

    // Reduction operators

    // Shrink the tensor of given dim 1.
    Tensor shrink_at(int dim1);

    Tensor tile(vector<int> dimens);

    static Tensor sum(Tensor &ts, vector<int> dims);

    friend Tensor operator+(Tensor &t1, Tensor &t2);

    friend Tensor operator+(Tensor &t1, VariantData t2);

    friend Tensor operator-(Tensor &t1, VariantData a);

    friend Tensor operator-(Tensor &t1, Tensor &t2);

    friend Tensor operator*(Tensor &t1, Tensor &t2);

    friend Tensor operator*(Tensor &t1, VariantData t2);


    friend Tensor operator==(Tensor &t1, Tensor &t2);

    friend Tensor operator!=(Tensor &t1, Tensor &t2);

    friend Tensor operator<=(Tensor &t1, Tensor &t2);

    friend Tensor operator>=(Tensor &t1, Tensor &t2);

    friend Tensor operator<(Tensor &t1, Tensor &t2);

    friend Tensor operator>(Tensor &t1, Tensor &t2);

    Tensor inv_pt() throw();

    void set_node(shared_ptr<grad::Node> node);
};

Tensor sum(Tensor &ts, int dim, bool keepdim);

Tensor mean(Tensor &ts, int dim, bool keepdim);

Tensor max(Tensor &ts, int dim, bool keepdim);

Tensor min(Tensor &ts, int dim, bool keepdim);


// Math operators
//static Tensor add(Tensor&t1, VariantData vd)throw();
Tensor apply(Tensor &ts, double(*fn)(double));

Tensor apply_two(Tensor &ts, Tensor &ts2, double(*fn)(double, double));

Tensor Sin(Tensor &ts);

Tensor Sin_no_grad(ts::Tensor &ts);

Tensor Cos(Tensor &ts);

Tensor Cos_no_grad(Tensor &ts);


Tensor Exp(ts::Tensor &ts);

Tensor Exp_no_grad(ts::Tensor &ts);

Tensor Ln(ts::Tensor &ts);

Tensor Ln_no_grad(ts::Tensor &ts);

Tensor Pow(Tensor &ts, unsigned int n);

Tensor inv_pt(Tensor ts) throw();

Tensor add(Tensor &t1, Tensor &t2) throw();

Tensor add(Tensor &t1, VariantData &t2) throw();

Tensor add_with_grad(Tensor &t1, Tensor &t2) throw();

Tensor add_no_grad(Tensor const &t1, Tensor const &t2) throw();

Tensor sub(Tensor &t1, Tensor &t2) throw();

Tensor sub(Tensor &t1, VariantData &t2) throw();

Tensor sub_with_grad(Tensor &t1, Tensor &t2) throw();

Tensor sub_no_grad(Tensor const &t1, Tensor const &t2) throw();

Tensor mul_pt(Tensor &t1, Tensor &t2) throw();

Tensor mul_pt(Tensor &t1, VariantData &t2) throw();

Tensor mul_pt_with_grad(Tensor &t1, Tensor &t2) throw();

Tensor mul_pt_no_grad(Tensor const &t1, Tensor const &t2) throw();

Tensor div_pt(Tensor &t1, Tensor &t2) throw();

Tensor div_pt(Tensor &t1, VariantData &t2) throw();

Tensor div_pt_with_grad(Tensor &t1, Tensor &t2) throw();

Tensor div_pt_no_grad(Tensor const &t1, Tensor const &t2) throw();

Tensor eq(Tensor &t1, Tensor &t2) throw();

Tensor ne(Tensor &t1, Tensor &t2) throw();

Tensor ge(Tensor &t1, Tensor &t2) throw();

Tensor gt(Tensor &t1, Tensor &t2) throw();

Tensor le(Tensor &t1, Tensor &t2) throw();

Tensor lt(Tensor &t1, Tensor &t2) throw();


template<typename T, size_t N>
static Tensor rand(int(&size)[N]){
    Tensor t = Tensor::init_with_shape<T>(size, N);
    std::random_device rd;  // 获取随机数种子
    std::mt19937 gen(rd()); // 初始化Mersenne Twister伪随机数生成器
    std::uniform_real_distribution<> distrib(0, 100);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) distrib(gen);
    }
    return t;
}

template<typename T>
static Tensor rand(int *arr, const int dim){
    int size[dim];
    for(int i = 0; i < dim; i++){
        size[i] = arr[i];
    }
    Tensor t = Tensor::init_with_shape<T>(size, dim);
    std::random_device rd;  // 获取随机数种子
    std::mt19937 gen(rd()); // 初始化Mersenne Twister伪随机数生成器
    std::uniform_real_distribution<> distrib(0, 100);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) distrib(gen);
    }
    return t;
}

template<typename T>
static Tensor rand(vector<int> shape){
    int shape_arr[shape.size()];
    for(int i = 0; i < shape.size(); i++){
        shape_arr[i] = shape[i];
    }
    return rand<T>(shape_arr, shape.size());
}

template<typename T, size_t N>
static Tensor zeros(int(&size)[N]){
    Tensor t = Tensor::init_with_shape<T>(size, N);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) 0;
    }
    return t;
}

template<typename T>
static Tensor zeros(vector<int> shape){
    int shape_arr[shape.size()];
    for(int i = 0; i < shape.size(); i++){
        shape_arr[i] = shape[i];
    }
    return zeros<T>(shape_arr, shape.size());
}

template<typename T>
static Tensor zeros(int *arr, const int dim){
    int size[dim];
    for(int i = 0; i < dim; i++){
        size[i] = arr[i];
    }
    Tensor t = Tensor::init_with_shape<T>(size, dim);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) 0;
    }
    return t;
}

static Tensor zeros(int dtype_id, int *arr, const int dim){
    switch(dtype_id){
        case 0:
            return zeros<bool>(arr, dim);
        case 1:
            return zeros<int>(arr, dim);
        case 2:
            return zeros<float>(arr, dim);
        case 3:
            return zeros<double>(arr, dim);
    }
}


size_t coordinates_to_index(vector<size_t> coordinates, int *shape, int dim);

size_t coordinates_to_index_with_fixed_dim(
        vector<size_t> coordinates, int *shape, int dim, int fixed_dim);

size_t *shape_to_acc(int *shape, int dim);

vector<size_t> index_to_coordinates(size_t index, int *shape, int dim);

vector<Tensor> subtensors_at_dim(Tensor &ts, int subdim);

/*
    zeros_like with exact type.
*/

//esin-sum-begin

Tensor einsum(char *a, Tensor &t1, Tensor &t2) throw();

Tensor einsum(char *a, Tensor &t1) throw();

Tensor einisum(std::string_view pattern, std::pair<Tensor &, Tensor &> inputs);

//esin-sum-end

static Tensor zeros_like(Tensor const &ts){
    switch(ts.get_dtype_id()){
        case 0:
            return zeros<double>(ts.get_shape(), ts.get_dimension());
        case 1:
            return zeros<int>(ts.get_shape(), ts.get_dimension());
        case 2:
            return zeros<float>(ts.get_shape(), ts.get_dimension());
        case 3:
            return zeros<double>(ts.get_shape(), ts.get_dimension());
    }

    return Tensor();
}

template<typename T, size_t N>
static Tensor ones(int(&size)[N]){
    Tensor t = Tensor::init_with_shape(size);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) 1;
    }
    return t;
}

template<typename T>
static Tensor ones(vector<int> shape){
    int shape_arr[shape.size()];
    for(int i = 0; i < shape.size(); i++){
        shape_arr[i] = shape[i];
    }
    return ones<T>(shape_arr, shape.size());
}

template<typename T>
static Tensor ones(int *size, const int dim){
    Tensor t = Tensor::init_with_shape<T>(size, dim);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) 1;
    }
    return t;
}


static Tensor ones_like(Tensor const &ts){
    switch(ts.get_dtype_id()){
        case 0:
            return ones<double>(ts.get_shape(), ts.get_dimension());
        case 1:
            return ones<int>(ts.get_shape(), ts.get_dimension());
        case 2:
            return ones<float>(ts.get_shape(), ts.get_dimension());
        case 3:
            return ones<double>(ts.get_shape(), ts.get_dimension());
    }
}


static Tensor eye_like(Tensor &ts){
    switch(ts.get_dtype_id()){
        case 0:
            return Tensor::eye<double>(ts.get_shape(), ts.get_dimension());
        case 1:
            return Tensor::eye<int>(ts.get_shape(), ts.get_dimension());
        case 2:
            return Tensor::eye<float>(ts.get_shape(), ts.get_dimension());
        case 3:
            return Tensor::eye<double>(ts.get_shape(), ts.get_dimension());
    }
}


template<typename T, size_t N>
static Tensor full(int(&size)[N], T value){
    Tensor t = Tensor::init_with_shape(size);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T) value;
    }
    return t;
}

Tensor cat(const std::pair<Tensor, Tensor> &tensors, int dim);

// Expand the data to the given same dimension shape arr.
template<size_t N>
static Tensor tileto(Tensor &tensor, int(&dims)[N]){
    const int dimension = tensor.get_dimension();
    if(dimension != N)
        throw std::invalid_argument("dimension of tensor and dims should be equal");
    int size[dimension];
    for(int i = 0; i < dimension; i++){
        size[i] = tensor.get_shape()[i] * dims[i];
//        std::cout << size[i] << std::endl;
    }
    Tensor t = Tensor::init_with_shape(size, dimension);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = Tensor::copy_tile((tensor.data_ptr()), &t, i, tensor.get_shape(), 0);
    }
    return t;
}

static Tensor tileto(Tensor &tensor, int *dims, int N){
    const int dimension = tensor.get_dimension();
    if(dimension != N)
        throw std::invalid_argument("dimension of tensor and dims should be equal");
    int size[dimension];
    for(int i = 0; i < dimension; i++){
        size[i] = tensor.get_shape()[i] * dims[i];
//        std::cout << size[i] << std::endl;
    }
    Tensor t = Tensor::init_with_shape(size, dimension);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = Tensor::copy_tile((tensor.data_ptr()), &t, i, tensor.get_shape(), 0);
    }
    return t;
}

static Tensor tileto(Tensor &ts, vector<int> new_shape){
    int shape_arr[new_shape.size()];
    for(int i = 0; i < new_shape.size(); i++){
        shape_arr[i] = new_shape[i];
    }
    return tileto(ts, shape_arr, ts.get_dimension());
}

static Tensor tile(Tensor &ts, vector<int> dimens){
    int _dim = dimens.size();
    vector<int> rst_shape;
    if(ts.get_dimension() < _dim){
        while(ts.get_dimension() < _dim){
            rst_shape.push_back(1);
        }
        for(auto x: ts.get_shape_vec()){
            rst_shape.push_back(x);
        }
    }else if(ts.get_dimension() > _dim){
        while(rst_shape.size() + dimens.size() < ts.get_dimension()){
            rst_shape.push_back(1);
        }
        for(auto x: dimens){
            rst_shape.push_back(x);
        }
    }else{
        for(auto x: dimens){
            rst_shape.push_back(x);
        }
    }
    // Expand tensor.
    return ts::tileto(ts, rst_shape);
}


Tensor transpose(Tensor tensor, int dim1, int dim2);

Tensor permute(Tensor &tensor, int dim[]);

Tensor permute(Tensor &tensor, vector<int> &dims);

template<size_t N>
static Tensor view(Tensor &tensor, int(&shape)[N]){
    return tensor.view(shape);
}

}
