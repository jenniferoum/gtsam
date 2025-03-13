//
// Created by darshan on 3/11/25.
//

#ifndef G_H
#define G_H


#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <vector>

using namespace gtsam;

/**
 * Symmetry group (SO(3) |x so(3)) x SO(3) x ... x SO(3)
 * Each element of the B list is associated with a calibration state
 */
class G {
public:
    Rot3 A;                 // First SO(3) element
    Matrix3 a;              // so(3) element (skew-symmetric matrix)
    std::vector<Rot3> B;    // List of SO(3) elements for calibration
    
    /**
     * Initialize the symmetry group G
     * @param A SO3 element
     * @param a so(3) element (skew symmetric matrix)
     * @param B list of SO3 elements
     */
    G(const Rot3& A = Rot3::Identity(),
       const Matrix3& a = Matrix3::Zero(),
       const std::vector<Rot3>& B = std::vector<Rot3>());
    
    /**
     * Define the group operation (multiplication)
     * @param other Another group element
     * @return The product of this and other
     */
    G operator*(const G& other) const;
    
    /**
     * Return the inverse element of the symmetry group
     * @return The inverse of this group element
     */
    G inv() const;
    
    /**
     * Return the identity of the symmetry group
     * @param n Number of calibration elements
     * @return The identity element with n calibration components
     */
    static G identity(int n);
    
    /**
     * Return a group element X given by X = exp(x)
     * @param x Vector representation of Lie algebra element
     * @return Group element given by the exponential of x
     */
    static G exp(const Vector& x);
};
#endif //G_H
