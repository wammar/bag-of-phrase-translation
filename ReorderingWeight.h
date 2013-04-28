#ifndef _REORDERING_WEIGHT_H_
#define _REORDERING_WEIGHT_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>
#include <sstream>

#include <fst/fstlib.h>
#include <fst/weight.h>
#include <fst/util.h>

#include <bitset>

#define __INFINITY 300.0
#define __PRECISION 0.0005
    

using namespace fst;
using namespace std;

namespace fst {
  class ReorderingWeight : LogWeightTpl<double> {
    
  public:
    
    double nlogprob;
    unsigned bitvector;
    
    typedef ReorderingWeight Weight;
    typedef int Label;
    typedef int StateId;
    
    ReorderingWeight() {
    }
    
    ReorderingWeight(double nlogprob, unsigned bitvector) {
      this->nlogprob = nlogprob;
      this->bitvector = bitvector;
    }
    
    ReorderingWeight(const ReorderingWeight &x) {
      this->nlogprob = x.nlogprob;
      this->bitvector = x.bitvector;
    }
    
    bool Member() const {
      return nlogprob >= 0.0;
    }
    
    static const ReorderingWeight One() {
      ReorderingWeight tbr;
      tbr.nlogprob = 0.0;
      tbr.bitvector = 0;
      return tbr;
    }
    
    static const ReorderingWeight Zero() {
      ReorderingWeight tbr;
      tbr.nlogprob = __INFINITY;
      tbr.bitvector = ~(unsigned (0));
      return tbr;
    }
    
    static const std::string &Type() {
      static const string type = "ReorderingWeight";
      return type;
    }
    
    ReorderingWeight Quantize(float delta = __PRECISION) const {
      ReorderingWeight tbr(*this);
      tbr.nlogprob = (floor(tbr.nlogprob/delta + 0.5F) * delta);
      return tbr;
    }
    
    static uint64 Properties() {
      return kLeftSemiring | kRightSemiring | kCommutative | kPath | kIdempotent;
    }
    
    ReorderingWeight Reverse() const {
      return *this;
    }
        
    static ReorderingWeight NoWeight() {
      ReorderingWeight tbr;
      tbr.nlogprob = -1.0;
      return tbr;
    }
    
    /*
    std::ostream &operator<<(ostream &strm, const ReorderingWeight &x) {
      strm << "nlogprob=" << x.nlogprob << ", bitvector=";
      union temp {
        unsigned x;
        std::bitset<32> y;
      } copy;
      copy.x = x.bitvector;
      for(unsigned i = 0; i < 32; ++i) {
        strm << copy.y[i];
      }
      return strm.str();
    } 
    
    std::istream &operator>>(std::istream &strm, ReorderingWeight &w) 
      void operator>>(std::string text) {
      assert(false);
      // TODO: to be implemented 
    }
    */
    
    std::istream &Read(std::istream &f) {
      assert(false);
      // TODO: to be implemented
    }
    
    std::ostream Write(std::ostream &f) const {
      assert(false);
      // TODO: to be implemented
    }
    
    size_t Hash() const {
      assert(false);
      // TODO: to be implemented 
    }
    
    bool ApproxEqual(const ReorderingWeight &x, const ReorderingWeight &y, float delta=__PRECISION) const {
      return nlogprob <= x.nlogprob + delta && nlogprob >= x.nlogprob - delta;
    }
    
    bool operator!=(const ReorderingWeight &x) const {
      return !ApproxEqual(x, *this);
    }

    bool operator==(const ReorderingWeight &x) const {
      return ApproxEqual(x, *this);
    }
    
    static ReorderingWeight Divide(const ReorderingWeight &z, 
                                     const ReorderingWeight &y, 
                                     DivideType typ = DIVIDE_ANY) {
      ReorderingWeight tbr;
      tbr.nlogprob = z.nlogprob - y.nlogprob;
      tbr.bitvector = z.bitvector - y.bitvector;
      return tbr;
    }
    
    typedef ReorderingWeight ReverseWeight;
    
  }; // end of ReorderingWeight
  
  static ReorderingWeight Plus(const ReorderingWeight &x, 
                               const ReorderingWeight &y) {
    assert((x.nlogprob >= 0) && (y.nlogprob >= 0));
    if(x.nlogprob <= y.nlogprob) {
      return x;
      } else {
      return y;
    }
  }
  
  static ReorderingWeight Times(const ReorderingWeight &x, 
                                const ReorderingWeight &y) {
    ReorderingWeight tbr;
    tbr.bitvector = x.bitvector | y.bitvector;
    // compatibility check
    if( (x.bitvector & y.bitvector) == 0) {
      tbr.nlogprob = __INFINITY;
    } else {
      tbr.nlogprob = x.nlogprob + y.nlogprob;
    }
    return tbr;
  }
  
  struct ReorderingArc {
    typedef ReorderingWeight Weight;
    typedef int Label;
    typedef int StateId;
   
    static const std::string &Type() {
      static const string type = "ReorderingArc";
      return type;
    }
    ReorderingArc(Label ilabel, Label olabel, Weight weight, StateId nextstate) : ilabel(ilabel), olabel(olabel), weight(weight), nextstate(nextstate) { }
    ReorderingArc() { ilabel = olabel = nextstate = 0; weight = Weight::Zero(); }
    ~ReorderingArc() { }
    
    //const static std::string type;
    Label ilabel;
    Label olabel;
    Weight weight;
    StateId nextstate;
  }; // end of ReorderingArc
} // end of namespace fst

#endif
