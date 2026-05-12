#pragma once

#include <utility>

template<typename T>
class DoubleBuf {
    public:
        template <typename... U>
        DoubleBuf(U&&... vals)
             : read_a_write_b(true),
               a(std::forward<U>(vals)...),
               b(std::forward<U>(vals)...) {}
        
        void swap() { read_a_write_b ^= true; }

        T& read() { return read_a_write_b ? a : b; }
        T& write() { return read_a_write_b ? b : a; }
    
    private:
        DoubleBuf() {} // No public default constructor

        bool read_a_write_b;
        T a, b;
};