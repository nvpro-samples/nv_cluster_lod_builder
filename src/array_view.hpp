//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

/**
 * @file array_view.hpp
 * @brief Defines ArrayView, an extended pointer that holds a size and byte stride.
 *
 * The primary motivation is to provide an API where the data may come directly from many sources, but with stronger
 * type and array bounds safety. For example:
 *
 * @code
 * std::vector<int> src1{1, 2, 3};
 * struct Obj { int i; double d; };
 * Obj src2[]{{3, 0.0}, {4, 0.0}, {5, 0.0}};
 * apiFunc(ArrayView(src1));
 * apiFunc(ArrayView(&src2[0].i, 3, sizeof(Obj)));
 * @endcode
 *
 * Type conversions and slicing also helps keep track of indices.
 *
 * @code
 * struct Obj { int x, y; };
 * Obj src[]{{1, 2}, {2, 4}, {3, 9}};
 * ArrayView view(src);
 * ArrayView<int> cast(view);
 * apiFunc(cast.slice(2, 4));  // passes an int array of {2, 4, 3, 9}
 * @endcode
 *
 * The DynamicArrayView object introduces a resize callback so an API can write varying amounts of data.
 *
 * @code
 * std::vector<int> src{1, 2, 3};
 * DynamicArrayView view(src);
 * view.resize(5, 42);  // src now has {1, 2, 3, 42, 42}
 * @endcode
 */

#pragma once

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#if defined(_NDEBUG)
#define ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION 0
#else
#define ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION 1
#endif

#if ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION
#define ARRAY_VIEW_BOUNDS_CHECK(expr) assert(expr)
#else
#define ARRAY_VIEW_BOUNDS_CHECK(expr) static_cast<void>(0)
#endif

/**
 * @brief Basic pointer iterator for ArrayView, but with a byte stride
 */
template <class ValueType>
class StrideIterator
{
public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = ValueType;
  using pointer           = value_type*;
  using reference         = value_type&;
  using difference_type   = ptrdiff_t;
  using stride_type       = size_t;
  using byte_pointer      = std::conditional_t<std::is_const_v<value_type>, const uint8_t*, uint8_t*>;

#if ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION
  using size_type = uint64_t;
  StrideIterator(pointer ptr, stride_type stride, size_type size)
      : m_ptr(ptr)
      , m_stride(stride)
      , m_begin(ptr)
      , m_end(reinterpret_cast<pointer>(reinterpret_cast<byte_pointer>(ptr) + size * stride))
  {
  }
#else
  StrideIterator(pointer ptr, stride_type stride)
      : m_ptr(ptr)
      , m_stride(stride)
  {
  }
#endif
  StrideIterator()                            = default;
  StrideIterator(const StrideIterator& other) = default;

  bool operator==(const StrideIterator& other) const { return (m_ptr == other.m_ptr); }
  bool operator!=(const StrideIterator& other) const { return (m_ptr != other.m_ptr); }
  bool operator<(const StrideIterator& other) const { return (m_ptr < other.m_ptr); }
  bool operator<=(const StrideIterator& other) const { return (m_ptr <= other.m_ptr); }
  bool operator>(const StrideIterator& other) const { return (m_ptr > other.m_ptr); }
  bool operator>=(const StrideIterator& other) const { return (m_ptr >= other.m_ptr); }

  StrideIterator& operator+=(const difference_type& i)
  {
    m_ptr = reinterpret_cast<pointer>(reinterpret_cast<byte_pointer>(m_ptr) + static_cast<size_t>(i) * m_stride);
    return (*this);
  }
  StrideIterator& operator-=(const difference_type& i)
  {
    m_ptr = reinterpret_cast<pointer>(reinterpret_cast<byte_pointer>(m_ptr) - i * m_stride);
    return (*this);
  }
  StrideIterator& operator++()
  {
    m_ptr = reinterpret_cast<pointer>(reinterpret_cast<byte_pointer>(m_ptr) + m_stride);
    return (*this);
  }
  StrideIterator& operator--()
  {
    m_ptr = reinterpret_cast<pointer>(reinterpret_cast<byte_pointer>(m_ptr) - m_stride);
    return (*this);
  }
  StrideIterator operator++(int)
  {
    auto result(*this);
    ++(*this);
    return result;
  }
  StrideIterator operator--(int)
  {
    auto result(*this);
    --(*this);
    return result;
  }
  StrideIterator operator+(const difference_type& i) const
  {
    StrideIterator result(*this);
    return result += i;
  }
  StrideIterator operator-(const difference_type& i) const
  {
    StrideIterator result(*this);
    return result -= i;
  }

  difference_type operator-(const StrideIterator& other) const
  {
    const auto& lhs = reinterpret_cast<const byte_pointer&>(m_ptr);
    const auto& rhs = reinterpret_cast<const byte_pointer&>(other.m_ptr);
    return static_cast<difference_type>((lhs - rhs) / m_stride);
  }

  value_type& operator*()
  {
    ARRAY_VIEW_BOUNDS_CHECK(m_ptr >= m_begin && m_ptr < m_end);
    return *m_ptr;
  }
  const value_type& operator*() const
  {
    ARRAY_VIEW_BOUNDS_CHECK(m_ptr >= m_begin && m_ptr < m_end);
    return *m_ptr;
  }
  value_type* operator->()
  {
    ARRAY_VIEW_BOUNDS_CHECK(m_ptr >= m_begin && m_ptr < m_end);
    return m_ptr;
  }

  // Relative offset accessor. Used by e.g. std::reduce().
  value_type& operator[](difference_type idx) const { return *(*this + idx); }

private:
  pointer     m_ptr{nullptr};
  stride_type m_stride{0};

#if ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION
  pointer m_begin{nullptr};
  pointer m_end{nullptr};
#endif
};

/**
 * @brief Random access container view - just (pointer, size, stride).
 *
 * - Constructable from a std::vector
 * - Supports implicit casting between types, e.g. uint[3] -> uvec[1]
 * - More type and size safety when making these conversions
 *
 * Similar to a C++20 std::span, but with a stride.
 */
template <class ValueType>
class ArrayView
{
public:
  using value_type  = ValueType;
  using size_type   = size_t;
  using stride_type = size_t;
  using iterator    = StrideIterator<value_type>;

  // Constructs an empty view
  ArrayView()
      : m_ptr(nullptr)
      , m_size(0)
      , m_stride(sizeof(value_type))  // default stride for iterating an empty view
  {
  }

  // Construct from std::vector. Use remove_cv_t as a vector's type cannot be
  // const but the vector can be
  ArrayView(std::vector<std::remove_cv_t<value_type>>& vector)
      : m_ptr(vector.data())
      , m_size(vector.size())
      , m_stride(sizeof(value_type))  // tightly packed
  {
  }

  // Const std::vector version
  ArrayView(const std::vector<std::remove_cv_t<value_type>>& vector)
      : m_ptr(vector.data())  // if passing a const vector, make sure you have ArrayView<const ...>
      , m_size(vector.size())
      , m_stride(sizeof(value_type))  // tightly packed
  {
  }

  // Disallow r-value const reference std::vector construction
  ArrayView(const std::vector<std::remove_cv_t<value_type>>&& vector) = delete;

  // Simple pointer + size wrapper, but keeping type safety
  ArrayView(value_type* ptr, size_type size, stride_type stride = sizeof(value_type))
      : m_ptr(ptr)
      , m_size(size)
      , m_stride(stride)
  {
    ARRAY_VIEW_BOUNDS_CHECK(stride > 0);
  }

  ArrayView(const ArrayView& other) = default;

  // The assignment operator copies pointers, not data!
  ArrayView& operator=(const ArrayView& other) = default;

  // Implicit conversion to ConstArrayView, i.e. ArrayView<const T>. Only exists for constructing a const value type
  // from a non const value type. The template parameter Dummy is used to avoid defining a copy constructor and breaking
  // the rule of 5. Based on https://quuxplusone.github.io/blog/2018/12/01/const-iterator-antipatterns/
  template <bool Dummy = true, class = std::enable_if_t<Dummy && std::is_const_v<ValueType>>>
  ArrayView(const ArrayView<std::remove_const_t<ValueType>>& other)
      : m_ptr(other.data())
      , m_size(other.size())
      , m_stride(other.stride())
  {
  }

  /**
  * @brief Constructor to convert from a different type of ArrayView object
  *
  * Marked explicit because this is somewhat dangerous, e.g. it can hide vec4 to vec3 conversion and not even assert if
  * the sizes make it an even multiple
  */
  template <class T,
            class = std::enable_if_t<
                // Must be converting from a different type
                !std::is_same_v<std::decay_t<ValueType>, std::decay_t<T>> &&
                // And not a const conversion (not sure why gcc was allowing reinterpret_cast without this)
                !(!std::is_const_v<value_type> && std::is_const_v<T>)>>
  explicit ArrayView(const ArrayView<T>& other)
      : m_ptr(reinterpret_cast<value_type*>(other.data()))  // const to non-const is a common error here. make sure you have ArrayView<const ...>
      , m_size((other.size() * static_cast<size_type>(sizeof(T))) / static_cast<size_type>(sizeof(value_type)))
      , m_stride(sizeof(T) == sizeof(value_type) ? other.stride() : static_cast<size_type>(sizeof(value_type)))
  {
    // Sanity check that both views now refer to the same amount of data
    ARRAY_VIEW_BOUNDS_CHECK(size() * static_cast<size_type>(sizeof(value_type))
                            == other.size() * static_cast<size_type>(sizeof(T)));
    ARRAY_VIEW_BOUNDS_CHECK(size() * stride() == other.size() * other.stride());

    // Either the array was tightly packed or the element size must be the same, to keep the same stride
    ARRAY_VIEW_BOUNDS_CHECK(sizeof(value_type) == sizeof(T) || other.stride() == static_cast<size_type>(sizeof(T)));
  }

  value_type& operator[](size_type idx) const
  {
    return *(begin() + static_cast<typename iterator::difference_type>(idx));
  }

  bool        empty() const { return m_size == 0; }
  value_type* data() const { return m_ptr; }
  size_type   size() const { return m_size; }
  stride_type stride() const { return m_stride; }

#if ARRAY_VIEW_ITERATOR_OVERFLOW_DETECTION
  iterator begin() const { return iterator(m_ptr, m_stride, m_size); }
  iterator end() const { return iterator(m_ptr, m_stride, m_size) + m_size; }
#else
  iterator begin() const { return iterator(m_ptr, m_stride); }
  iterator end() const { return iterator(m_ptr, m_stride) + m_size; }
#endif

  ArrayView slice(size_type position, size_type length) const
  {
    ARRAY_VIEW_BOUNDS_CHECK(position < m_size);
    ARRAY_VIEW_BOUNDS_CHECK(length <= m_size - position);
    return ArrayView(&*(begin() + position), length, m_stride);
  }

  // Returns a slice if the view is not empty, otherwise, returns an empty view
  ArrayView slice_nonempty(size_type position, size_type length) const
  {
    return empty() ? ArrayView{} : slice(position, length);
  }

protected:
  value_type* m_ptr;
  size_type   m_size;
  stride_type m_stride;
};

// Deduction guides for constructing a VectorView with a std::vector, necessary for const vectors.
template <class VectorValueType>
ArrayView(std::vector<VectorValueType>& vector) -> ArrayView<VectorValueType>;
template <class VectorValueType>
ArrayView(const std::vector<VectorValueType>& vector) -> ArrayView<const VectorValueType>;

// Utility type to force the value type to be const or non-const
template <class ValueType>
using ConstArrayView = ArrayView<const ValueType>;
template <class ValueType>
using MutableArrayView = ArrayView<std::remove_const_t<ValueType>>;

// Const to non const cast function, ideally never to be used.
template <class ValueType>
const ArrayView<ValueType>& ArrayViewConstCast(const ConstArrayView<ValueType>& constArrayView)
{
  return (const ArrayView<ValueType>&)(constArrayView);
}

static_assert(std::is_constructible_v<ConstArrayView<int>, ArrayView<int>>);
static_assert(!std::is_constructible_v<ArrayView<int>, ConstArrayView<int>>);
static_assert(std::is_copy_constructible_v<ConstArrayView<int>>);
static_assert(std::is_trivially_copy_constructible_v<ConstArrayView<int>>);
static_assert(std::is_copy_constructible_v<ArrayView<int>>);
static_assert(std::is_trivially_copy_constructible_v<ArrayView<int>>);
static_assert(std::is_constructible_v<ConstArrayView<int>, const std::vector<int>&>);
static_assert(!std::is_constructible_v<ConstArrayView<int>, const std::vector<int>&&>,
              "Creating an ArrayView from an r-value would make a dangling pointer");

// TODO get rid of this

/**
 * @brief Adds a resize callback to ArrayView
 */
template <class ValueType>
class DynamicArrayView : public ArrayView<ValueType>
{
public:
  using typename ArrayView<ValueType>::value_type;
  using typename ArrayView<ValueType>::size_type;
  using typename ArrayView<ValueType>::stride_type;
  using resize_func_type = ValueType*(size_type, const ValueType&);

  DynamicArrayView()                              = default;
  DynamicArrayView(const DynamicArrayView& other) = default;
  DynamicArrayView(std::function<resize_func_type> resizeCallback, value_type* ptr, size_type size, stride_type stride = sizeof(value_type))
      : ArrayView<ValueType>(ptr, size, stride)
      , m_resizeCallback(resizeCallback)
  {
  }

  // Implementation for std::vector, keeping a reference to the original container in the lambda's capture
  DynamicArrayView(std::vector<std::remove_cv_t<ValueType>>& vector)
      : ArrayView<ValueType>(vector)
      , m_resizeCallback([&vector](size_type size, const ValueType& value) {
        vector.resize(size, value);
        return vector.data();
      })
  {
  }

  // Type conversion constructor. Currently implemented by chaining lambdas, each encoding size and offset manipulation
  // of the original resize function. Alternatives would be to use void pointer return type and maintain size ratio and
  // offset variables.
  template <class T, class = std::enable_if_t<!std::is_same_v<value_type, T>>>
  DynamicArrayView(const DynamicArrayView<T>& other)
      : ArrayView<ValueType>(other)
      , m_resizeCallback([cb = other.m_resizeCallback](size_type size, const ValueType& value) {
        size_t otherTypeSize = static_cast<size_type>(sizeof(T));
        size_t thisTypeSize  = static_cast<size_type>(sizeof(value_type));
        return reinterpret_cast<value_type*>(cb((size * thisTypeSize) / otherTypeSize, reinterpret_cast<const T&>(value)));
      })
  {
  }

  void resize(size_type size, const ValueType& value = ValueType())
  {
    this->m_ptr  = m_resizeCallback(size, value);
    this->m_size = size;
    assert(!size || this->m_ptr);
  }

  // Returns true if this object has been initialized with a resize callback
  bool resizable() const { return static_cast<bool>(m_resizeCallback); }

private:
  std::function<resize_func_type> m_resizeCallback;

  // Provide access to m_resizeCallback from other DynamicArrayView types
  template <class T>
  friend class DynamicArrayView;
};
