// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/mechanics/quaternion_orientation.hpp>
#include <openpfc/runtime/gpu/quaternion_view.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

using namespace pfc::finite_strain;
namespace {
using Quat = std::array<double, 4>;
Quat axis_angle(double x, double y, double z, double a) {
  const double norm = std::sqrt(x*x+y*y+z*z);
  const double v = std::sin(a/2)/norm;
  return {std::cos(a/2), x*v, y*v, z*v};
}
Quat multiply(const Quat &a,const Quat &b) {
  return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
          a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
          a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
          a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
}
// Independent quaternion vector-rotation reference: R*v = q*(0,v)*conj(q).
Tensor2 passive_reference(const Quat &q) {
  const Quat inverse{q[0],-q[1],-q[2],-q[3]};
  Tensor2 Q;
  for(int axis=0;axis<3;++axis) {
    Quat e{0,0,0,0};e[axis+1]=1;
    const Quat output=multiply(multiply(q,e),inverse);
    for(int component=0;component<3;++component)
      Q(axis,component)=output[component+1];
  }
  return Q;
}
void compare(const Tensor2 &a,const Tensor2 &b,double tol=2e-13) {
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    REQUIRE(std::abs(a(i,j)-b(i,j)) < tol);
}
}

TEST_CASE("FCC quaternion frame matches an independent passive rotation",
          "[finite_strain][crystal][quaternion]") {
  const auto q=axis_angle(1,2,-3,1.13);
  Tensor2 Q;
  const double data[]{q[0],q[1],q[2],q[3]};
  REQUIRE(quaternion_sample_to_crystal(data,Q));
  compare(Q,passive_reference(q));
  const double neg[]{-q[0],-q[1],-q[2],-q[3]};
  Tensor2 equivalent;
  REQUIRE(quaternion_sample_to_crystal(neg,equivalent));
  compare(Q,equivalent,1e-14);
  const double right[]{std::sqrt(.5),0.,0.,std::sqrt(.5)};
  REQUIRE(quaternion_sample_to_crystal(right,Q));
  REQUIRE(std::abs(Q(0,1)-1.)<1e-14);
  REQUIRE(std::abs(Q(1,0)+1.)<1e-14);
}

TEST_CASE("FCC quaternion rejects undefined orientation instead of identity",
          "[finite_strain][crystal][quaternion]") {
  Tensor2 Q;
  const double zero[]{0,0,0,0};
  const double wrong[]{2,0,0,0};
  const double nan[]{std::numeric_limits<double>::quiet_NaN(),0,0,1};
  REQUIRE_FALSE(quaternion_sample_to_crystal(zero,Q));
  REQUIRE_FALSE(quaternion_sample_to_crystal(wrong,Q));
  REQUIRE_FALSE(quaternion_sample_to_crystal(nan,Q));
}

TEST_CASE("Padded quaternion field maps only the local owned brick",
          "[finite_strain][crystal][quaternion]") {
  constexpr std::size_t nx=3,ny=2,nz=4,h=2;
  std::array<std::vector<double>,4> buffers;
  FccQuaternionFieldView fields{};
  for(int c=0;c<4;++c) {
    buffers[c].resize((nx+2*h)*(ny+2*h)*(nz+2*h),-999.);
    fields.q[c]=buffers[c].data();
  }
  fields.owned[0]=nx;fields.owned[1]=ny;fields.owned[2]=nz;
  fields.padded[0]=nx+2*h;fields.padded[1]=ny+2*h;fields.padded[2]=nz+2*h;
  fields.storage_halo=h;
  REQUIRE(fields.valid_for(nx*ny*nz));
  for(std::size_t z=0;z<nz;++z)
    for(std::size_t y=0;y<ny;++y)
      for(std::size_t x=0;x<nx;++x) {
        const auto p=x+nx*(y+ny*z);
        const auto expected=(x+h)+(nx+2*h)*((y+h)+(ny+2*h)*(z+h));
        REQUIRE(fields.at(p)==expected);
      }
  fields.padded[1]--;
  REQUIRE_FALSE(fields.valid_for(nx*ny*nz));
  fields.padded[1]++;
  fields.q[1]=nullptr;
  REQUIRE_FALSE(fields.valid_for(nx*ny*nz));
}

TEST_CASE("Quaternion and existing dense FCC material-point results agree",
          "[finite_strain][crystal][quaternion]") {
  const FccSlipParameters p{2,1,.5,.001,5,2.25,.1,.02,.2,.05,1,1.4};
  const auto state=fcc_identity_state(p);
  Tensor2 F=identity2();F(0,0)=1.003;F(0,1)=.002;F(1,2)=.001;
  for(const auto &q:std::array<Quat,3>{Quat{1,0,0,0},
                                       axis_angle(1,2,3,.8),
                                       axis_angle(0,0,1,std::numbers::pi/2)}) {
    const double data[]{q[0],q[1],q[2],q[3]};
    Tensor2 converted;
    REQUIRE(quaternion_sample_to_crystal(data,converted));
    const auto reference=passive_reference(q);
    const auto a=try_evaluate_fcc(p,converted,state,F);
    const auto b=try_evaluate_fcc(p,reference,state,F);
    REQUIRE(a.status==FccStatus::Success);
    REQUIRE(b.status==FccStatus::Success);
    compare(a.value.update.piola,b.value.update.piola,1e-10);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      for(int k=0;k<3;++k)for(int l=0;l<3;++l)
        REQUIRE(std::abs(a.value.tangent(i,j,k,l)-b.value.tangent(i,j,k,l))<1e-9);
    for(int i=0;i<12;++i)
      REQUIRE(std::abs(a.value.update.state.shear[i]-b.value.update.state.shear[i])<1e-10);
  }
}

TEST_CASE("All 24 proper cubic quaternion symmetries preserve material response",
          "[finite_strain][crystal][quaternion]") {
  const FccSlipParameters p{2,1,.5,.001,5,2.25,.1,.02,.2,.05,1,1.4};
  const auto state=fcc_identity_state(p);
  Tensor2 F=identity2();F(0,0)=1.0005;F(0,1)=.0005;F(2,1)=.0003;
  const auto baseline=try_evaluate_fcc(p,identity2(),state,F);
  REQUIRE(baseline.status==FccStatus::Success);
  std::vector<Quat> quaternions{{1,0,0,0}};
  for(const auto &axis:std::array<std::array<double,3>,3>{
        std::array<double,3>{1,0,0},{0,1,0},{0,0,1}})
    for(double k: {0.5,1.,1.5})
      quaternions.push_back(axis_angle(axis[0],axis[1],axis[2],
                                       k*std::numbers::pi));
  for(const auto &axis:std::array<std::array<double,3>,4>{
        std::array<double,3>{1,1,1},{1,1,-1},
        {1,-1,1},{-1,1,1}})
    for(double k: {2./3.,4./3.})
      quaternions.push_back(axis_angle(axis[0],axis[1],axis[2],
                                       k*std::numbers::pi));
  for(const auto &axis:std::array<std::array<double,3>,6>{
        std::array<double,3>{1,1,0},{1,-1,0},
        {1,0,1},{1,0,-1},{0,1,1},{0,1,-1}})
    quaternions.push_back(axis_angle(axis[0],axis[1],axis[2],
                                     std::numbers::pi));
  REQUIRE(quaternions.size()==24);
  for(const auto &q:quaternions) {
    const double data[]{q[0],q[1],q[2],q[3]};
    Tensor2 Q;
    REQUIRE(quaternion_sample_to_crystal(data,Q));
    REQUIRE(crystal::proper_rotation(Q));
    const auto response=try_evaluate_fcc(p,Q,state,F);
    REQUIRE(response.status==FccStatus::Success);
    compare(response.value.update.piola,baseline.value.update.piola,5e-9);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      for(int k=0;k<3;++k)for(int l=0;l<3;++l)
        REQUIRE(std::abs(response.value.tangent(i,j,k,l)-
                         baseline.value.tangent(i,j,k,l))<5e-8);
  }
}
