//
// Created by darshan on 3/11/25.
//
#include "G.h"
#include "utilities.h"
#include <stdexcept>

G::G(const Rot3& A, const Matrix3& a, const std::vector<Rot3>& B)
    : A(A), a(a), B(B) {}

G G::operator*(const G& other) const {
    if (B.size() != other.B.size()) {
        throw std::invalid_argument("Group elements must have the same number of calibration elements");
    }
    
    std::vector<Rot3> new_B;
    for (size_t i = 0; i < B.size(); i++) {
        new_B.push_back(B[i] * other.B[i]);
    }
    
    return G(A * other.A,
            a + Rot3::Hat(A.matrix() * Rot3::Vee(other.a)),
            new_B);
}

G G::inv() const {
    Matrix3 A_inv = A.inverse().matrix();
    
    std::vector<Rot3> B_inv;
    for (const auto& b : B) {
        B_inv.push_back(b.inverse());
    }
    
    return G(A.inverse(), 
            -Rot3::Hat(A_inv * Rot3::Vee(a)),
            B_inv);
}

G G::identity(int n) {
    std::vector<Rot3> B(n, Rot3::Identity());
    return G(Rot3::Identity(), Matrix3::Zero(), B);
}

G G::exp(const Vector& x) {
    if (x.size() < 6 || x.size() % 3 != 0) {
        throw std::invalid_argument("Wrong size, a vector with size multiple of 3 and at least 6 must be provided");
    }
    
    int n = (x.size() - 6) / 3;
    Rot3 A = Rot3::Expmap(x.head<3>());
    
    Vector3 a_vee = Rot3::ExpmapDerivative(-x.head<3>()) * x.segment<3>(3);
    Matrix3 a = Rot3::Hat(a_vee);
    
    std::vector<Rot3> B;
    for (int i = 0; i < n; i++) {
        B.push_back(Rot3::Expmap(x.segment<3>(6 + 3*i)));
    }
    
    return G(A, a, B);
}