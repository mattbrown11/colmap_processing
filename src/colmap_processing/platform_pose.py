#!/usr/bin/env python
"""
ckwg +31
Copyright 2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import threading
from scipy.interpolate import interp1d


lock = threading.Lock()


class PlatformPoseProvider(object):
    """Camera manager object.

    This object supports requests for the navigation coordinate system state,
    at a particular time, relative to a local east/north/up coordinate system.

    Attributes:
    :param lat0: Latitude of the origin (deg).
    :type lat0: float

    :param lon0: Longitude of the origin (deg).
    :type lon0: float

    :param h0: Height above the WGS84 ellipsoid of the origin (meters).
    :type h0: float

    """
    def __init__(self):
        raise NotImplementedError

    def pos(self, t):
        """
        :param t: Time at which to query the platform state (time in seconds
            since Unix epoch).
        :type t: float

        :return: Position of the platform coordinate system relative to a
            local level east/north/up coordinate system.
        :rtype: 3-pos

        """
        raise NotImplementedError

    def quat(self, t):
        """
        :param t: Time at which to query the platform state (time in seconds
            since Unix epoch).
        :type t: float

        :return: Quaternion (qx, qy, qz, qw) specifying the orientation of the
            navigation coordinate system relative to a local level
            east/north/up (ENU) coordinate system. The quaternion represent a
            coordinate system rotation from ENU to the navigation coordinate
            system.
        :rtype: 4-array

        """
        raise NotImplementedError

    def pose(self, t):
        """
        :param t: Time at which to query the INS state (time in seconds since
            Unix epoch).
        :type t: float

        :return: List with the first element the 3-array position (see pos) and
            the second element the 4-array orientation quaternion (see quat).
        :rtype: list

        """
        raise NotImplementedError

    def __str__(self):
        return str(type(self))

    def __repr__(self):
        return self.__str__()


class PlatformPoseInterp(PlatformPoseProvider):
    """Interpolated pose from a time series.

    """
    def __init__(self, lat0=None, lon0=None, h0=None):
        """

        """
        self._pose_time_series = np.zeros((0,8))
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0

    @property
    def pose_time_series(self):
        with lock:
            return self._pose_time_series

    def add_to_pose_time_series(self, t, pos, quat):
        """Adds to pose time series such that time is monotonically increasiing.

        """
        pose = np.hstack([t,pos,quat])
        with lock:
            k = len(self._pose_time_series)
            if k == 0:
                self._pose_time_series = np.insert(self._pose_time_series, 0,
                                                   pose, axis=0)
                return

            while True:
                if self._pose_time_series[k-1,0] < t:
                    self._pose_time_series = np.insert(self._pose_time_series,
                                                       k, pose, axis=0)
                    break
                k -= 1
                if k == 0:
                    self._pose_time_series = np.insert(self._pose_time_series,
                                                       k, pose, axis=0)
                    break

            if len(self._pose_time_series) > 1e4:
                self._pose_time_series = np.delete(self._pose_time_series, 0,
                                                   0)

            if False:
                # Sort
                ind = np.argsort(self._pose_time_series[:,0])
                self._pose_time_series = self._pose_time_series[ind,:]

    def pose(self, t):
        """See PlatformPose documentation.

        """
        with lock:
            if False:
                # Check that times are sorted.
                ind = np.argsort(self._pose_time_series[:,0])
                assert np.all(np.diff(ind) == 1)

            min_ind = 0
            max_ind = len(self._pose_time_series)-1
            #print('max time:', self._pose_time_series[max_ind,0])
            if t <= self._pose_time_series[min_ind,0]:
                pose = self._pose_time_series[min_ind,1:]
            elif t >= self._pose_time_series[max_ind,0]:
                if False:
                    print('t:', t)
                    print('self._pose_time_series[max_ind,0]:',
                          self._pose_time_series[max_ind,0])
                    print('Time error:',
                          self._pose_time_series[max_ind,0]-t)

                pose = self._pose_time_series[max_ind,1:]
            else:
                try:
                    f = interp1d(self._pose_time_series[:,0],
                                 self._pose_time_series[:,1:], kind='linear',
                                 axis=0, bounds_error=True)
                    pose = f(t)
                except ValueError:
                    print('Available times:', self._pose_time_series[:,0])
                    print('Could not evaluate at time', t)
                    raise

            #print('Time:', t, 'pose:', pose)
            return [pose[:3], pose[3:]]

    def pos(self, t):
        """See PlatformPose documentation.

        """
        return self.pose(t)[0]
        #return self.average_pos()

    def quat(self, t):
        """See PlatformPose documentation.

        """
        return self.pose(t)[1]
        #return self.average_quat()

    def average_quat(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,4:], 0)

    def average_pos(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,1:4], 0)


class PlatformPoseFixed(PlatformPoseProvider):
    def __init__(self, pos=np.array([0,0,0]),
                 quat=np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]), lat0=None,
                 lon0=None, h0=None):
        self._pos = pos
        self._quat = quat
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0

    def pos(self, t):
        """See PlatformPose documentation.

        """
        return self._pos

    def quat(self, t):
        """See PlatformPose documentation.

        """
        return self._quat

    def pose(self, t):
        """See PlatformPose documentation.

        """
        return [self._pos, self._quat]
