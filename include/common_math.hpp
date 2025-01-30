#pragma once

#include "common.hpp"

namespace UTIL
{

#if __cplusplus >= 202002L // C++20 required

// Complex type checker for complex type
template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

// Concept for arithmetic types
template <typename T>
concept ArithmeticType = std::is_arithmetic<T>::value;

// Concepts for different types of comparison
template <typename T>
concept LessThanComparable =
    std::equality_comparable<T> && requires( const T lhs, const T rhs ) {
        { lhs < rhs } -> std::convertible_to<bool>;
    };

template <typename T>
concept MoreThanComparable =
    std::equality_comparable<T> && requires( const T lhs, const T rhs ) {
        { lhs > rhs } -> std::convertible_to<bool>;
    };

template <typename T>
concept Comparable = LessThanComparable<T> && MoreThanComparable<T>;

#elif __cplusplus >= 202302L // C++23 required

#endif

} // namespace UTIL
