// I really like this header for angles lol
 
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/



#ifndef GEOMETRY_ANGLES_UTILS_H
#define GEOMETRY_ANGLES_UTILS_H

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>

namespace angles
{

  static inline double from_degrees(double degrees)
  {
    return degrees * M_PI / 180.0;
  }

  static inline double to_degrees(double radians)
  {
    return radians * 180.0 / M_PI;
  }


  static inline double normalize_angle_positive(double angle)
  {
    const double result = fmod(angle, 2.0*M_PI);
    if(result < 0) return result + 2.0*M_PI;
    return result;
  }


  static inline double normalize_angle(double angle)
  {
    const double result = fmod(angle + M_PI, 2.0*M_PI);
    if(result <= 0.0) return result + M_PI;
    return result - M_PI;
  }


  static inline double shortest_angular_distance(double from, double to)
  {
    return normalize_angle(to-from);
  }

  static inline double two_pi_complement(double angle)
  {
    //check input conditions
    if (angle > 2*M_PI || angle < -2.0*M_PI)
      angle = fmod(angle, 2.0*M_PI);
    if(angle < 0)
      return (2*M_PI+angle);
    else if (angle > 0)
      return (-2*M_PI+angle);

    return(2*M_PI);
  }

  static bool find_min_max_delta(double from, double left_limit, double right_limit, double &result_min_delta, double &result_max_delta)
  {
    double delta[4];

    delta[0] = shortest_angular_distance(from,left_limit);
    delta[1] = shortest_angular_distance(from,right_limit);

    delta[2] = two_pi_complement(delta[0]);
    delta[3] = two_pi_complement(delta[1]);

    if(delta[0] == 0)
    {
      result_min_delta = delta[0];
      result_max_delta = std::max<double>(delta[1],delta[3]);
      return true;
    }

    if(delta[1] == 0)
    {
        result_max_delta = delta[1];
        result_min_delta = std::min<double>(delta[0],delta[2]);
        return true;
    }


    double delta_min = delta[0];
    double delta_min_2pi = delta[2];
    if(delta[2] < delta_min)
    {
      delta_min = delta[2];
      delta_min_2pi = delta[0];
    }

    double delta_max = delta[1];
    double delta_max_2pi = delta[3];
    if(delta[3] > delta_max)
    {
      delta_max = delta[3];
      delta_max_2pi = delta[1];
    }


    //    printf("%f %f %f %f\n",delta_min,delta_min_2pi,delta_max,delta_max_2pi);
    if((delta_min <= delta_max_2pi) || (delta_max >= delta_min_2pi))
    {
      result_min_delta = delta_max_2pi;
      result_max_delta = delta_min_2pi;
      if(left_limit == -M_PI && right_limit == M_PI)
        return true;
      else
        return false;
    }
    result_min_delta = delta_min;
    result_max_delta = delta_max;
    return true;
  }


  static inline bool shortest_angular_distance_with_large_limits(double from, double to, double left_limit, double right_limit, double &shortest_angle)
  {
    // Shortest steps in the two directions
    double delta = shortest_angular_distance(from, to);
    double delta_2pi = two_pi_complement(delta);

    // "sort" distances so that delta is shorter than delta_2pi
    if(std::fabs(delta) > std::fabs(delta_2pi))
      std::swap(delta, delta_2pi);

    if(left_limit > right_limit) {
      // If limits are something like [PI/2 , -PI/2] it actually means that we
      // want rotations to be in the interval [-PI,PI/2] U [PI/2,PI], ie, the
      // half unit circle not containing the 0. This is already gracefully
      // handled by shortest_angular_distance_with_limits, and therefore this
      // function should not be called at all. However, if one has limits that
      // are larger than PI, the same rationale behind shortest_angular_distance_with_limits
      // does not hold, ie, M_PI+x should not be directly equal to -M_PI+x.
      // In this case, the correct way of getting the shortest solution is to
      // properly set the limits, eg, by saying that the interval is either
      // [PI/2, 3*PI/2] or [-3*M_PI/2, -M_PI/2]. For this reason, here we
      // return false by default.
      shortest_angle = delta;
      return false;
    }

    // Check in which direction we should turn (clockwise or counter-clockwise).

    // start by trying with the shortest angle (delta).
    double to2 = from + delta;
    if(left_limit <= to2 && to2 <= right_limit) {
      // we can move in this direction: return success if the "from" angle is inside limits
      shortest_angle = delta;
      return left_limit <= from && from <= right_limit;
    }

    // delta is not ok, try to move in the other direction (using its complement)
    to2 = from + delta_2pi;
    if(left_limit <= to2 && to2 <= right_limit) {
      // we can move in this direction: return success if the "from" angle is inside limits
      shortest_angle = delta_2pi;
      return left_limit <= from && from <= right_limit;
    }

    // nothing works: we always go outside limits
    shortest_angle = delta; // at least give some "coherent" result
    return false;
  }


  static inline bool shortest_angular_distance_with_limits(double from, double to, double left_limit, double right_limit, double &shortest_angle)
  {

    double min_delta = -2*M_PI;
    double max_delta = 2*M_PI;
    double min_delta_to = -2*M_PI;
    double max_delta_to = 2*M_PI;
    bool flag    = find_min_max_delta(from,left_limit,right_limit,min_delta,max_delta);
    double delta = shortest_angular_distance(from,to);
    double delta_mod_2pi  = two_pi_complement(delta);


    if(flag)//from position is within the limits
    {
      if(delta >= min_delta && delta <= max_delta)
      {
        shortest_angle = delta;
        return true;
      }
      else if(delta_mod_2pi >= min_delta && delta_mod_2pi <= max_delta)
      {
        shortest_angle = delta_mod_2pi;
        return true;
      }
      else //to position is outside the limits
      {
        find_min_max_delta(to,left_limit,right_limit,min_delta_to,max_delta_to);
          if(fabs(min_delta_to) < fabs(max_delta_to))
            shortest_angle = std::max<double>(delta,delta_mod_2pi);
          else if(fabs(min_delta_to) > fabs(max_delta_to))
            shortest_angle =  std::min<double>(delta,delta_mod_2pi);
          else
          {
            if (fabs(delta) < fabs(delta_mod_2pi))
              shortest_angle = delta;
            else
              shortest_angle = delta_mod_2pi;
          }
          return false;
      }
    }
    else // from position is outside the limits
    {
        find_min_max_delta(to,left_limit,right_limit,min_delta_to,max_delta_to);

          if(fabs(min_delta) < fabs(max_delta))
            shortest_angle = std::min<double>(delta,delta_mod_2pi);
          else if (fabs(min_delta) > fabs(max_delta))
            shortest_angle =  std::max<double>(delta,delta_mod_2pi);
          else
          {
            if (fabs(delta) < fabs(delta_mod_2pi))
              shortest_angle = delta;
            else
              shortest_angle = delta_mod_2pi;
          }
      return false;
    }

    shortest_angle = delta;
    return false;
  }
}

#endif