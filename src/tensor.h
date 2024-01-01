#include <variant>
#include <iostream>
#include <type_traits>
#include <iomanip>
#include <vector>
#include <memory>
#include "random"

namespace ts {
    using VariantData = std::variant<float, int, double, bool>;

    class Tensor {
    private:
        std::shared_ptr<VariantData[]> data_shared;
        VariantData *data;
        int dimension;
        std::shared_ptr<int[]> shape;
        int total_size;
        std::string type_name;
        template<typename ArrayType, size_t N>
        void copyData(ArrayType (&arr)[N], VariantData *&dest, VariantData * destEnd)  {
            for (size_t i = 0; i < N; ++i) {
                if constexpr (std::is_array<ArrayType>::value) {
                    copyData(arr[i], dest, destEnd);
                } else {
                    *dest = arr[i];
                    if (dest < destEnd)
                        dest++;
                }
            }
        }

        template<typename ArrayType, size_t N>
        void getShape(ArrayType (&arr)[N], int dim) {
            shape.get()[dim] = N;
            if constexpr (std::is_array<ArrayType>::value)
                getShape(arr[0], dim + 1);
        }

        static VariantData copy_tile(VariantData *src, Tensor *dst, int idx, int *src_shape, int dim);

        void print(std::ostream& os, int index, int dim) const;


        template<size_t N>
        static Tensor getShapeTensor(int (&size)[N]) {
            Tensor t = Tensor();
            t.dimension = N;
            t.shape.reset(new int[t.dimension]);
            t.total_size = 1;
            for (int i = 0; i < t.dimension; i++) {
                t.shape[i] = size[i];
                t.total_size *= size[i];
            }
            return t;
        }

    public:
        int* size();

        std::string type();

        VariantData *data_ptr();

        template<typename T, size_t N>
        explicit Tensor(T (&arr)[N]) {
            // dimension of arr, e.g. double[2][1] dimension = 2
            dimension = std::rank<T>::value + 1;
            shape.reset(new int[dimension]);
            using BaseType = typename std::remove_all_extents<T>::type();
            type_name = typeid(BaseType).name();

            getShape(arr, 0);

            // copy array to data
            total_size = 1;
            for (int i = 0; i < dimension; i++) {
                total_size *= shape[i];
            }
            data_shared.reset(new VariantData[total_size]);
            data = data_shared.get();
            VariantData *pointer = data;
            copyData(arr, pointer, data_shared.get() + total_size);
        }

        Tensor();

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);



        template<typename T, size_t N>
        static Tensor zeros(int (&size)[N]) {
            Tensor t = getShapeTensor(size);
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = (T)0;
            }
            return t;
        }

        template<typename T, size_t N>
        static Tensor ones(int (&size)[N]) {
            Tensor t = getShapeTensor(size);
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = (T)1;
            }
            return t;
        }

        template<typename T, size_t N>
            static Tensor full(int (&size)[N], T value) {
            Tensor t = getShapeTensor(size);
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = (T)value;
            }
            return t;
        }

        template<typename T, size_t N>
        static Tensor rand(int (&size)[N])  {
            Tensor t = getShapeTensor(size);
            std::random_device rd;  // 获取随机数种子
            std::mt19937 gen(rd()); // 初始化Mersenne Twister伪随机数生成器
            std::uniform_real_distribution<> distrib(0, 100);
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = (T)distrib(gen);
            }
            return t;
        }

        template<typename T, size_t N>
        static Tensor eye(int (&size)[N]) {
            Tensor t = getShapeTensor(size);
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = (T)0;
            }
            int min = t.shape[0] < t.shape[1] ? t.shape[0] : t.shape[1];
            for (int i = 0; i < min; i++) {
                t.data[i * t.shape[1] + i] = (T)1;
            }
            return t;
        }

        static Tensor cat(const std::pair<Tensor, Tensor>& tensors, int dim);

        template<size_t N>
        static Tensor tile(const Tensor& tensor, int (&dims)[N]) {
            Tensor t = Tensor();
            t.dimension = tensor.dimension;
            t.shape.reset(new int[t.dimension]);
            for (int i = 0; i < t.dimension; i++) {
                t.shape[i] = tensor.shape[i] * dims[i];
            }
            t.total_size = 1;
            for (int i = 0; i < t.dimension; i++) {
                t.total_size *= t.shape[i];
            }
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                t.data[i] = copy_tile((tensor.data), &t, i, tensor.shape.get(), 0);
            }
            return t;
        }

        Tensor operator()(int idx);


        Tensor operator()(int idx, std::pair<int, int> range);

        void operator=(const VariantData& value);

        template<size_t N>
        void operator=(const int (&arr)[N]) {
            copyData(arr, data, data + total_size);
        }

        Tensor transpose(int dim1, int dim2);

        static Tensor transpose(Tensor& tensor, int dim1, int dim2);

        void init_data(Tensor &t, int size) {

        }


    };


}

