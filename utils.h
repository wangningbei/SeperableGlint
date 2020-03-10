#ifndef __SREF_UTILS_H
#define __SREF_UTILS_H

#define RNG_ROUNDS     8
#define USE_VECTORIZED 1
#define ANISOTROPY_LIMIT 4

#include <mitsuba/core/qmc.h>
#include "gaussian.h"

MTS_NAMESPACE_BEGIN

enum EIntersectionResult {
	EDisjoint = 0,
	EIntersection,
	EInside,
	EContained,
};

/// Helper function: intersect a 2D AABB with a line segment
inline bool intersectAABBLineSegment(const AABB2 &aabb, Point2 p0, Point2 p1) {
	/* Check if all AABB points are to one side of the line */

	Float last = 0;
	bool allOnSameSide = true;

	for (int i=0; i<5; ++i) {
		Float x = ((i+1) & 2) ? aabb.min[0] : aabb.max[0],
			  y = (i     & 2) ? aabb.min[1] : aabb.max[1],
			  v = (p1.y-p0.y) * x + (p0.x-p1.x) * y + (p1.x*p0.y - p0.x * p1.y);
		if (last*v < 0 || (i > 0 && last*v == 0))
			allOnSameSide = false;
		last = v;
	}

	if (allOnSameSide)
		return false;

	/* Check if the X/Y-projection of the segment and bounding box overlap */
	for (int i=0; i<2; ++i) {
		if (p1[i] < p0[i])
			std::swap(p0[i], p1[i]);
		if (aabb.min[i] < p0[i]) {
			if (p0[i] > aabb.max[i])
				return false;
		} else {
			if (aabb.min[i] > p1[i])
				return false;
		}
	}

	return true;
}

/// Helper function: Intersection between planar line segments -- doesn't handle the collinear case
bool lineIntersection(const Point2 &p0, const Point2 &p1, const Point2 &p2, const Point2 &p3, Point2 &p) {
	Vector2 s1 = p1-p0, s2 = p3-p2;

	Float s = (-s1.y * (p0.x - p2.x) + s1.x * (p0.y - p2.y)) / (-s2.x * s1.y + s1.x * s2.y);
	Float t = ( s2.x * (p0.y - p2.y) - s2.y * (p0.x - p2.x)) / (-s2.x * s1.y + s1.x * s2.y);

	p = p0 + t*s1;

	return s>=0 && s<=1 && t>=0 && t<=1;
}

/// Polygon clipping function: axis aligned case
template <typename PointType> size_t sutherlandHodgman(PointType *input, size_t inCount, PointType *output, int axis, Float splitPos, bool isMinimum) {
	if (inCount < 3)
		return 0;

	PointType cur     = input[0];
	Float sign        = isMinimum ? 1.0f : -1.0f;
	Float distance    = sign * (cur[axis] - splitPos);
	bool  curIsInside = (distance >= 0);
	size_t outCount   = 0;

	for (size_t i=0; i<inCount; ++i) {
		size_t nextIdx = i+1;
		if (nextIdx == inCount)
			nextIdx = 0;
		PointType next = input[nextIdx];
		distance = sign * (next[axis] - splitPos);
		bool nextIsInside = (distance >= 0);

		if (curIsInside && nextIsInside) {
			/* Both this and the next vertex are inside, add to the list */
			output[outCount++] = next;
		} else if (curIsInside && !nextIsInside) {
			/* Going outside -- add the intersection */
			Float t = (splitPos - cur[axis]) / (next[axis] - cur[axis]);
			PointType p = cur + (next - cur) * t;
			p[axis] = splitPos; // Avoid roundoff errors
			output[outCount++] = p;
		} else if (!curIsInside && nextIsInside) {
			/* Coming back inside -- add the intersection + next vertex */
			Float t = (splitPos - cur[axis]) / (next[axis] - cur[axis]);
			PointType p = cur + (next - cur) * t;
			p[axis] = splitPos; // Avoid roundoff errors
			output[outCount++] = p;
			output[outCount++] = next;
		} else {
			/* Entirely outside - do not add anything */
		}
		cur = next;
		curIsInside = nextIsInside;
	}
	return outCount;
}

/// Polygon clipping function: non axis aligned case
template <typename PointType> size_t sutherlandHodgman(PointType *input, size_t inCount, PointType *output, const Point2 &p0, const Point2 &p1) {
	if (inCount < 3)
		return 0;

	Vector2 n(p1.y-p0.y, p0.x-p1.x);

	PointType cur     = input[0];
	bool  curIsInside = dot(cur-p0, n) <= 0;
	size_t outCount   = 0;

	for (size_t i=0; i<inCount; ++i) {
		size_t nextIdx = i+1;
		if (nextIdx == inCount)
			nextIdx = 0;
		PointType next = input[nextIdx];
		bool nextIsInside = dot(next-p0, n) <= 0;

		if (curIsInside && nextIsInside) {
			output[outCount++] = next;
		} else if (curIsInside && !nextIsInside) {
			Point2 p;
			lineIntersection(cur, next, p0, p1, p);
			output[outCount++] = p;
		} else if (!curIsInside && nextIsInside) {
			Point2 p;
			lineIntersection(cur, next, p0, p1, p);
			output[outCount++] = p;
			output[outCount++] = next;
		}
		cur = next;
		curIsInside = nextIsInside;
	}
	return outCount;
}

/// Helper function: check if two spherical arcs intersect
inline bool sphArcIntersect(const Vector &s1, const Vector &e1,
							const Vector &s2, const Vector &e2) {
	Vector v = cross(cross(s1, e1), cross(s2, e2));
	Float l2 = v.lengthSquared();
	if (l2 == 0)
		return false;

	v *= math::signum(v.z) / std::sqrt(l2);

	float dp1 = dot(s1, e1), dp1s = dot(s1, v), dp1e = dot(e1, v);
	float dp2 = dot(s2, e2), dp2s = dot(s2, v), dp2e = dot(e2, v);

	if ((dp1*dp1e - dp1s)*(dp1*dp1 - 1.f) <= 0 ||
		(dp1*dp1s - dp1e)*(dp1*dp1 - 1.f) <= 0 ||
		(dp2*dp2e - dp2s)*(dp2*dp2 - 1.f) <= 0 ||
		(dp2*dp2s - dp2e)*(dp2*dp2 - 1.f) <= 0)
		return false;

	return true;
}

struct Triangle2 {
	Point2 v0, v1, v2;

	inline Triangle2() { }

	inline Triangle2(const Point2 &v0, const Point2 &v1, const Point2 &v2)
		: v0(v0), v1(v1), v2(v2) { }

	void fixHandedness() {
		Vector2 a = v1-v0, b = v2-v0;
		if (a.x*b.y-a.y*b.x < 0)
			std::swap(v1, v2);
	}

	Float area() const {
		return std::abs(.5f * det(v1-v0, v2-v0));
	}

	AABB2 getAABB() const {
		AABB2 aabb(v0);
		aabb.expandBy(v1);
		aabb.expandBy(v2);
		return aabb;
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Triangle2[v0=" << v0.toString() << ", v1=" << v1.toString() << ", v2=" << v2.toString() << "]";
		return oss.str();
	}
};

struct Parallelogram2 {
	Point2 o;
	Vector2 v0, v1;
	AABB2 bbox;

	inline Parallelogram2() { }
	inline Parallelogram2(const Point2 &o, const Vector2 &v0, const Vector2 &v1)
		: o(o), v0(v0), v1(v1) { }

	void computeBoundingBox() {
		bbox = AABB2(o);
		bbox.expandBy(o+v0);
		bbox.expandBy(o+v1);
		bbox.expandBy(o+v0+v1);
	}

	inline Float area() const {
		return std::abs(det(v0, v1));
	}

	/// Check if a 2D parallelogram contains a given point
	inline bool contains(const Point2 &q) const {
		Vector2 rel = q - o;
		Float dc = det(v0, v1),
			  d0 = det(rel, v1),
			  d1 = det(v0, rel);

		if (dc < 0) {
			dc *= -1; d0 *= -1; d1 *= -1;
		}

		return d0 >= 0 && d0 <= dc && d1 >= 0 && d1 <= dc;
	}

	inline EIntersectionResult intersectAABB(const AABB2 &uv) const {
		if (!uv.overlaps(bbox)) {
			return EDisjoint;
		} else if (contains(uv.getCorner(0)) && contains(uv.getCorner(1)) && contains(uv.getCorner(2)) && contains(uv.getCorner(3))) {
			return EInside;
		} else {
			if (uv.contains(o) || uv.contains(o+v0) ||
				uv.contains(o+v1) || uv.contains(o+v0+v1)) {
				return EIntersection;
			} else if (intersectAABBLineSegment(uv, o, o + v0) ||
				intersectAABBLineSegment(uv, o + v0, o + v0 + v1) ||
				intersectAABBLineSegment(uv, o + v0 + v1, o + v1) ||
				intersectAABBLineSegment(uv, o + v1, o)) {
				return EIntersection;
			}
		}

		return EDisjoint;
	}

	inline Float overlapAABB(const TAABB<Point2> &aabb) const {
		const size_t maxVerts = 12;
		Point2 vertices1[maxVerts], vertices2[maxVerts];

		vertices1[0] = o;
		vertices1[1] = o+v0;
		vertices1[2] = o+v0+v1;
		vertices1[3] = o+v1;

		size_t nVertices = 4;
		for (int axis=0; axis<Point2::dim; ++axis) {
			nVertices = sutherlandHodgman(vertices1, nVertices, vertices2, axis, aabb.min[axis], true);
			nVertices = sutherlandHodgman(vertices2, nVertices, vertices1, axis, aabb.max[axis], false);
		}

		Float area = 0;
		if (nVertices >= 3) {
			for (size_t idx=0; idx<nVertices; ++idx) {
				size_t nextIdx = idx+1;
				if (nextIdx == nVertices)
					nextIdx = 0;
				const Point2 &cur = vertices1[idx];
				const Point2 &next = vertices1[nextIdx];

				area += cur.x * next.y - next.x * cur.y;
			}
		}

		return area*.5f;
	}

	inline Float overlapTriangle(const Triangle2 &tri) const {
		const size_t maxVerts = 12;
		Point2 vertices1[maxVerts], vertices2[maxVerts];

		size_t nVertices = 4;
		vertices1[0] = o;
		vertices1[1] = o+v0;
		vertices1[2] = o+v0+v1;
		vertices1[3] = o+v1;

		nVertices = sutherlandHodgman(vertices1, nVertices, vertices2, tri.v0, tri.v1);
		nVertices = sutherlandHodgman(vertices2, nVertices, vertices1, tri.v1, tri.v2);
		nVertices = sutherlandHodgman(vertices1, nVertices, vertices2, tri.v2, tri.v0);

		Float area = 0;
		if (nVertices >= 3) {
			for (size_t idx=0; idx<nVertices; ++idx) {
				size_t nextIdx = idx+1;
				if (nextIdx == nVertices)
					nextIdx = 0;
				const Point2 &cur = vertices2[idx];
				const Point2 &next = vertices2[nextIdx];

				area += cur.x * next.y - next.x * cur.y;
			}
		}

		return std::abs(area*.5f);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Parallelogram2[o=" << o.toString() << ", v0=" << v0.toString() << ", v1=" << v1.toString() << "]";
		return oss.str();
	}
};

/// Always-positive modulo function, single precision version (assumes b > 0)
inline float modulo(float a, float b) {
	float r = std::fmod(a, b);
	return (r < 0.0f) ? r + b : r;
}

/// Computes the overlap between a pos.oriented triangle in the plane and the unit circle at the origin
Float circleOverlap(Point2 points[3]) {
	struct ClassifiedPoint : public Point2 {
		inline ClassifiedPoint(const Vector2 &v, int edge, bool inside)
			: Point2(v), edge(edge), inside(inside) { }
		inline ClassifiedPoint() { }
		int edge;
		bool inside;
	};

	ClassifiedPoint cpoints[6];
	int ctr = 0, signs = 0;

	for (int edge=0; edge<3; ++edge) {
		Vector2 pc = Vector2(points[edge]), pn = Vector2(points[(edge+1)%3]);

		if (dot(pc, pc) < 1)
			cpoints[ctr++] = ClassifiedPoint(pc, edge, true);

		Vector2 pd = pn-pc;
		Float dp = dot(pc, pd);
		signs |= (dp > 0) ? 1 : ((dp < 0) ? 2 : 0);

		Float t0, t1;
		if (solveQuadratic(dot(pd,pd), 2*dp, dot(pc,pc)-1, t0, t1)) {
			if (t0 >= 0 && t0 <= 1)
				cpoints[ctr++] = ClassifiedPoint((1-t0) * pc + t0 * pn, edge, false);
			if (t1 >= 0 && t1 <= 1)
				cpoints[ctr++] = ClassifiedPoint((1-t1) * pc + t1 * pn, edge, false);
		}
	}

	if (ctr == 0 && (signs == 1 || signs == 2))
		return M_PI;

	Float sum = 0;
	for (int i=0; i<ctr; ++i) {
		const ClassifiedPoint &p0 = cpoints[i];
		const ClassifiedPoint &p1 = cpoints[(i+1)%ctr];
		bool wrap = (i == ctr-1);

		if (p0.inside || p1.inside || (p0.edge == p1.edge && !wrap))
			/* Linear segment */
			sum += .5f * (p0.x*p1.y - p0.y*p1.x);
		else
			/* Circular segment */
			sum += .5f * modulo(std::atan2(p1.y, p1.x) - std::atan2(p0.y, p0.x), 2*M_PI);
	}

	return std::abs(sum);
}

struct SphericalTriangle {
	size_t id;
	Vector v0, v1, v2;

	inline SphericalTriangle() { }
	inline SphericalTriangle(size_t id, const Vector &v0, const Vector &v1, const Vector &v2)
		: id(id), v0(v0), v1(v1), v2(v2) { }

	inline SphericalTriangle(int id) : id(id) {
		if (id == 1) {
			// "root" triangle
			v0 = v1 = v2 = Vector(0.0f);
		} else {
			const Vector root_vertices[5] = {
				Vector(0, 0, 1),
				Vector(1, 0, 0),
				Vector(0, 1, 0),
				Vector(-1, 0, 0),
				Vector(0, -1, 0)
			};

			const int root_triangles[4][3] = {
				{1, 0, 4},
				{4, 0, 3},
				{3, 0, 2},
				{2, 0, 1}
			};

			v0 = root_vertices[root_triangles[id-2][0]];
			v1 = root_vertices[root_triangles[id-2][1]];
			v2 = root_vertices[root_triangles[id-2][2]];
		}
	}

	void split(SphericalTriangle *ch) const {
		if (id != 1) {
			Vector s0 = normalize(v0 + v1),
				   s1 = normalize(v1 + v2),
				   s2 = normalize(v2 + v0);

			size_t nid = 4*id - 2;
			ch[0] = SphericalTriangle(nid,   s0, s1, s2);
			ch[1] = SphericalTriangle(nid+1, v0, s0, s2);
			ch[2] = SphericalTriangle(nid+2, s0, v1, s1);
			ch[3] = SphericalTriangle(nid+3, s2, s1, v2);
		} else {
			// "root" triangle
			for (int i=0; i<4; ++i)
				ch[i] = SphericalTriangle(i+2);
		}
	}

	void split(SphericalTriangle &ch0, SphericalTriangle &ch1, SphericalTriangle &ch2, SphericalTriangle &ch3) const {
		if (id != 1) {
			Vector s0 = normalize(v0 + v1),
				   s1 = normalize(v1 + v2),
				   s2 = normalize(v2 + v0);

			size_t nid = 4*id - 2;
			ch0 = SphericalTriangle(nid,   s0, s1, s2);
			ch1 = SphericalTriangle(nid+1, v0, s0, s2);
			ch2 = SphericalTriangle(nid+2, s0, v1, s1);
			ch3 = SphericalTriangle(nid+3, s2, s1, v2);
		} else {
			// "root" triangle
			ch0 = SphericalTriangle(2);
			ch1 = SphericalTriangle(3);
			ch2 = SphericalTriangle(4);
			ch3 = SphericalTriangle(5);
		}
	}
	void splitOneNode(SphericalTriangle &ch3) const {
		if (id != 1) {
			Vector s1 = normalize(v1 + v2),
				s2 = normalize(v2 + v0);

			size_t nid = 4 * id - 2;
			ch3 = SphericalTriangle(nid + 3, s2, s1, v2);
		}
		else {
			// "root" triangle
			ch3 = SphericalTriangle(5);
		}
	}
	inline Vector center() const {
		return normalize(v0+v1+v2);
	}

	inline Vector planarNormal() const {
		Vector side1(v1-v0), side2(v2-v0);
		return normalize(cross(side1, side2));
	}

	void fixHandedness() {
		if (dot(v0, cross(v1, v2)) < 0)
			std::swap(v1, v2);
	}

	inline bool isRoot() const {
		return id == 1;
	}

	inline Float areaAccurate() const {
		if (isRoot())
			return 2*M_PI;// "root" triangle

		Float a = unitAngle(v1, v2),
			  b = unitAngle(v2, v0),
			  c = unitAngle(v0, v1),
			  s = (a+b+c) / 2;

		return 4.0f * std::atan(math::safe_sqrt(
			std::tan(0.5f * s) *
			std::tan(0.5f * (s-a)) *
			std::tan(0.5f * (s-b)) *
			std::tan(0.5f * (s-c)))
		);
	}

	inline Float area() const {
		if (isRoot())
			return 2*M_PI;// "root" triangle
		Float a = .5f * cross(v1-v0, v2-v0).length();
		return a * (1.0f + a*(0.534479f + a*(0.00394938f + a*0.53551f)));
	}

	inline Vector &operator[](int i) {
		return ((Vector *) this)[i];
	}

	inline const Vector &operator[](int i) const {
		return ((Vector *) this)[i];
	}

	inline bool contains(const Vector &v) const {
		return (dot(v, cross(v0, v1)) > 0 &&
		        dot(v, cross(v1, v2)) > 0 &&
		        dot(v, cross(v2, v0)) > 0);
	}

	inline bool overlaps(const SphericalTriangle &tri) const {
		if (isRoot())
			return true;
		if (contains(tri.v0) || contains(tri.v1) || contains(tri.v2))
			return true;
		if (tri.contains(v0) || tri.contains(v1) || tri.contains(v2))
			return true;

		for (int i=0; i<3; ++i) {
			Vector s0 = operator[](i),
			       e0 = operator[]((i+1)%3);
			for (int j=0; j<3; ++j) {
				Vector s1 = tri[j],
				       e1 = tri[(j+1)%3];
				if (sphArcIntersect(s0, e0, s1, e1))
					return true;
			}
		}
		return false;
	}

	/// Return a string representation
	inline std::string toString() const {
		std::ostringstream oss;
		oss << "SphericalTriangle[v0=" << v0.toString()
			<< ", v1=" << v1.toString() << ", v2=" << v2.toString() << "]";
		return oss.str();
	}
};

struct SphericalConic {
	Matrix3x3 M;
	Vector center;
	bool ellipse;

	inline SphericalConic() { }

	inline SphericalConic(const Vector &wi, const Vector &wo, Float alpha, bool clamp = true, bool *clamped = NULL) {
		center = normalize(wi+wo);
		Vector v1 = normalize(wi-wo);
		Vector v2 = cross(v1, center);

		Matrix3x3 Q(v1, v2, center);

		Float cosAlpha = std::cos(alpha);

		Float mu = dot(wi, wo);
		Float cotAlphaH = 1/std::tan(alpha/2);

		Float ev[3] = {
			cotAlphaH*cotAlphaH,
			(mu + cosAlpha) / (1 - cosAlpha),
			-1
		};

		if (clamp) {
			Float cotLimit  = 1/std::tan(alpha/2 * ANISOTROPY_LIMIT),
				  cotLimit2 = cotLimit*cotLimit;
			if (ev[1] < cotLimit2) {
				ev[1] = cotLimit2;

				if (clamped)
					*clamped = true;
			}
		}

		ellipse = ev[1] > 0;

		M.setZero();
		for (int i=0; i<3; ++i)
			for (int j=0; j<3; ++j)
				for (int k=0; k<3; ++k)
					M(i, j) += Q(i, k) * Q(j, k) * ev[k];
	}

	inline SphericalConic(const Vector &wo, Float alpha) {
		center = normalize(wo);
		Vector v1, v2;
		coordinateSystem(center, v1, v2);

		Matrix3x3 Q(v1, v2, center);

		Float cotAlphaH = 1/std::tan(alpha);

		Float ev[3] = {
			cotAlphaH*cotAlphaH,
			cotAlphaH*cotAlphaH,
			-1
		};

		ellipse = true;

		M.setZero();
		for (int i=0; i<3; ++i)
			for (int j=0; j<3; ++j)
				for (int k=0; k<3; ++k)
					M(i, j) += Q(i, k) * Q(j, k) * ev[k];
	}

	inline Float eval(const Vector &v) const {
		return dot(v, M * v);
	}

	inline bool contains(const Vector &v) const {
		return dot(v, M * v) < 0;
	}

	inline bool intersectSphArc(const Vector &v0, const Vector &v1) const {
		Vector sum = v0 + v1, diff = v1 - v0;

		Float A = 0, B = 0, C = 0;
		for (int i=0; i<3; ++i) {
			for (int j=0; j<3; ++j) {
				const Float entry = M(i, j);
				A += entry * diff[i] * diff[j];
				B += entry * sum[i]  * diff[j];
				C += entry * sum[i]  * sum[j];
			}
		}
		B *= 2;

		Float x0, x1;
		if (!solveQuadratic(A, B, C, x0, x1))
			return false;

		return std::abs(x0) <= 1 || std::abs(x1) <= 1;
	}


	inline void _mm_debug_ps(const char *desc, __m128 value) const {
		float dest[4];
		_mm_storeu_ps(dest, value);
		printf("%s: [%f, %f, %f, %f]\n", desc, dest[0], dest[1], dest[2], dest[3]);
	}

	inline static __m128 _mm_sel_ps_xor(const __m128& a, const __m128& b, const __m128& mask) {
		return _mm_xor_ps( a, _mm_and_ps( mask, _mm_xor_ps( b, a ) ) );
	}

	inline static __m128 _mm_abs_ps(__m128 x) {
		static const __m128 sign_mask = _mm_set1_ps(-0.f);
		return _mm_andnot_ps(sign_mask, x);
	}

	EIntersectionResult intersectSphTriangleVectorized(const SphericalTriangle &tri) const {
		if (tri.isRoot())
			return EContained;

		static const __m128
			const0 = _mm_set1_ps(2.0f),
			const1 = _mm_set1_ps(4.0f),
			const2 = _mm_set1_ps(-0.5f);

		__m128
			x  = _mm_set_ps(0, tri.v2.x, tri.v1.x, tri.v0.x),
			y  = _mm_set_ps(0, tri.v2.y, tri.v1.y, tri.v0.y),
			z  = _mm_set_ps(0, tri.v2.z, tri.v1.z, tri.v0.z),
			sx = _mm_shuffle_ps(x, x, 201),
			sy = _mm_shuffle_ps(y, y, 201),
			sz = _mm_shuffle_ps(z, z, 201),
			sum_x  = _mm_add_ps(sx, x),
			diff_x = _mm_sub_ps(sx, x),
			sum_y  = _mm_add_ps(sy, y),
			diff_y = _mm_sub_ps(sy, y),
			sum_z  = _mm_add_ps(sz, z),
			diff_z = _mm_sub_ps(sz, z),
			A, B, C, D, entry;

		const Float *matrix = (const Float *) M.m;
		entry = _mm_set1_ps(matrix[0]);
		A = _mm_mul_ps(_mm_mul_ps(diff_x, diff_x), entry);
		B = _mm_mul_ps(_mm_mul_ps(sum_x,  diff_x), entry);
		C = _mm_mul_ps(_mm_mul_ps(sum_x,  sum_x ), entry);
		D = _mm_mul_ps(_mm_mul_ps(x,      x     ), entry);

		entry = _mm_set1_ps(matrix[1]);
		B = _mm_add_ps(B, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sum_x, diff_y),
			_mm_mul_ps(sum_y, diff_x)), entry));
		entry = _mm_mul_ps(entry, const0);
		A = _mm_add_ps(A, _mm_mul_ps(_mm_mul_ps(diff_x, diff_y), entry));
		C = _mm_add_ps(C, _mm_mul_ps(_mm_mul_ps(sum_x,  sum_y ), entry));
		D = _mm_add_ps(D, _mm_mul_ps(_mm_mul_ps(x,      y     ), entry));

		entry = _mm_set1_ps(matrix[2]);
		B = _mm_add_ps(B, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sum_x, diff_z),
			_mm_mul_ps(sum_z, diff_x)), entry));
		entry = _mm_mul_ps(entry, const0);
		A = _mm_add_ps(A, _mm_mul_ps(_mm_mul_ps(diff_x, diff_z), entry));
		C = _mm_add_ps(C, _mm_mul_ps(_mm_mul_ps(sum_x,  sum_z ), entry));
		D = _mm_add_ps(D, _mm_mul_ps(_mm_mul_ps(x,      z     ), entry));

		entry = _mm_set1_ps(matrix[4]);
		A = _mm_add_ps(A, _mm_mul_ps(_mm_mul_ps(diff_y, diff_y), entry));
		B = _mm_add_ps(B, _mm_mul_ps(_mm_mul_ps(sum_y,  diff_y), entry));
		C = _mm_add_ps(C, _mm_mul_ps(_mm_mul_ps(sum_y,  sum_y ), entry));
		D = _mm_add_ps(D, _mm_mul_ps(_mm_mul_ps(y,      y     ), entry));

		entry = _mm_set1_ps(matrix[5]);
		B = _mm_add_ps(B, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sum_y, diff_z),
			_mm_mul_ps(sum_z, diff_y)), entry));
		entry = _mm_mul_ps(entry, const0);
		A = _mm_add_ps(A, _mm_mul_ps(_mm_mul_ps(diff_y, diff_z), entry));
		C = _mm_add_ps(C, _mm_mul_ps(_mm_mul_ps(sum_y,  sum_z ), entry));
		D = _mm_add_ps(D, _mm_mul_ps(_mm_mul_ps(y,      z     ), entry));

		entry = _mm_set1_ps(matrix[8]);
		A = _mm_add_ps(A, _mm_mul_ps(_mm_mul_ps(diff_z, diff_z), entry));
		B = _mm_mul_ps(_mm_add_ps(B, _mm_mul_ps(_mm_mul_ps(sum_z, diff_z), entry)), const0);
		C = _mm_add_ps(C, _mm_mul_ps(_mm_mul_ps(sum_z,  sum_z ), entry));
		D = _mm_add_ps(D, _mm_mul_ps(_mm_mul_ps(z,      z     ), entry));

		int insideMask = _mm_movemask_ps(D);
		if (insideMask && insideMask != 7)
			return EIntersection;

		__m128 sqrtDiscrim = _mm_sqrt_ps(_mm_sub_ps(_mm_mul_ps(B, B),
			_mm_mul_ps(const1, _mm_mul_ps(A, C))));

		__m128 temp = _mm_mul_ps(const2,
			_mm_sel_ps_xor(_mm_add_ps(B, sqrtDiscrim), _mm_sub_ps(B, sqrtDiscrim),
			_mm_cmplt_ps(B, _mm_setzero_ps())));

		A = _mm_abs_ps(A);
		C = _mm_abs_ps(C);
		temp = _mm_abs_ps(temp);

		__m128 result = _mm_and_ps(_mm_cmplt_ps(temp, A), _mm_cmplt_ps(C, temp));
		int intersectMask = _mm_movemask_ps(result);
		if (intersectMask)
			return EIntersection;

		if (insideMask == 7)
			return EInside;

		__m128
			cx = _mm_sub_ps(_mm_mul_ps(y, sz), _mm_mul_ps(z, sy)),
			cy = _mm_sub_ps(_mm_mul_ps(z, sx), _mm_mul_ps(x, sz)),
			cz = _mm_sub_ps(_mm_mul_ps(x, sy), _mm_mul_ps(y, sx));

		__m128 dp = _mm_add_ps(_mm_mul_ps(cx, _mm_set1_ps(center.x)),
			_mm_add_ps(_mm_mul_ps(cy, _mm_set1_ps(center.y)),
			_mm_mul_ps(cz, _mm_set1_ps(center.z))));

		int containsMask = _mm_movemask_ps(dp);
		if (containsMask == 0)
			return EContained;

		return EDisjoint;
	}

	inline EIntersectionResult intersectSphTriangle(
			const SphericalTriangle &tri) const {
		if (tri.isRoot())
			return EContained;

		int insideCount =
			(contains(tri.v0) ? 1 : 0) +
			(contains(tri.v1) ? 1 : 0) +
			(contains(tri.v2) ? 1 : 0);

		if (insideCount > 0 && insideCount != 3)
			return EIntersection;

		else if (intersectSphArc(tri.v0, tri.v1) ||
			intersectSphArc(tri.v1, tri.v2) ||
			intersectSphArc(tri.v2, tri.v0))
			return EIntersection;

		else if (insideCount == 3)
			return EInside;

		else if (tri.contains(center))
			return EContained;

		return EDisjoint;
	}

	/// Compute the overlapping fraction with a spherical triangle -- only approximate
	inline Float sphTriangleOverlap(const SphericalTriangle &tri) const {
		Frame frame(tri.planarNormal());

		/* Convert conic section to an ellipse in 2D */
		Matrix3x3 frameQ(frame.s, frame.t, frame.n), frameQT;
		frameQ.transpose(frameQT);
		Matrix3x3 M = frameQT * this->M * frameQ;

		Float denom = M(0,1)*M(1,0) - M(0,0)*M(1,1);
		if (std::abs(denom) < RCPOVERFLOW)
			return 0.0f;

		/* Switch to coordinates where the ellipse center
		   is at the origin */
		Point2 center = Point2(
			M(1,1)*M(2,0) - M(1,0)*M(2,1),
			M(0,0)*M(2,1) - M(1,0)*M(2,0)) / denom;

		Float normalization = 1 / (-M(2,2)
				+   M(0,0)*center.x*center.x
				+   M(1,1)*center.y*center.y
				+ 2*M(0,1)*center.x*center.y),
			A = M(0,0)*normalization,
			B = M(0,1)*normalization,
			C = M(1,1)*normalization;

		/* Finally, re-scale the ellipse to a circle and
		   transform the spherical triangle into that space */
		Float f0 = math::safe_sqrt(4*B*B + (A-C)*(A-C)),
			  f1 = math::safe_sqrt(.5f*(A+C+f0)) / f0,
			  f2 = math::safe_sqrt(.5f*(A+C-f0)) / f0;

		Matrix2x2 trafo(
			.5f * (f2 * (C-A+f0) + (A-C+f0)*f1), B*(f1-f2),
			B*(f1 - f2),  .5f * ((A-C+f0)*f2 + (C-A+f0)*f1)
		);

		/* Now find the plane version of the spherical triangle */
		Vector v0 = frame.toLocal(tri.v0),
			   v1 = frame.toLocal(tri.v1),
			   v2 = frame.toLocal(tri.v2);

		Point2 points[3] = {
			Point2(trafo * (Vector2(v0.x, v0.y) / v0.z - Vector2(center))),
			Point2(trafo * (Vector2(v1.x, v1.y) / v1.z - Vector2(center))),
			Point2(trafo * (Vector2(v2.x, v2.y) / v2.z - Vector2(center)))
		};

		Vector2 a = points[1]-points[0], b = points[2]-points[0];
		Float triangleArea = .5f * (a.x*b.y-a.y*b.x);
		if (triangleArea < 0) {
			std::swap(points[1], points[2]);
			triangleArea = -triangleArea;
		}

		return circleOverlap(points) / triangleArea;
	}

	/// Return a string representation
	inline std::string toString() const {
		std::ostringstream oss;
		oss << "SphericalConic[" << endl
			<< "  M = " << indent(M.toString()) << "," << endl
			<< "  center = " << center.toString() << endl
			<< "]";
		return oss.str();
	}
};

/// Describes a query region modeled as a parallelogram in UV space, and an outgoing direction
struct QueryRegion {
	Parallelogram2 uv;
	SphericalConic dir;
	size_t nExpansions;

	inline QueryRegion(const Point2 &p, const Vector2 &dx, const Vector2 &dy,
		const Vector &wi, const Vector &wo, Float alpha) : uv(p, dx, dy),
		dir(wi, wo, alpha), nExpansions(0) {
	}


	inline QueryRegion(const Parallelogram2 &uv, const SphericalConic &dir) :
		uv(uv), dir(dir), nExpansions(0) { }

	/// Return a string representation
	inline std::string toString() const {
		std::ostringstream oss;
		oss << "QueryRegion[uv=" << uv.toString() << ", dir=" << dir.toString() << "]";
		return oss.str();
	}
};

void sampleMultinomialApprox(size_t _idx, float *p, ssize_t count, size_t *result) {
#if 0
	//this should be removed later
	int countA = 0;
	for (int i = 0; i < 4; i++)
	{
		result[i] = p[i] * count;
		countA += result[i];
	}
	int gap = count - countA;
	for (int i = 0; i < gap; i++)
		result[i] += 1;
	return;
#endif

	const int rounds = RNG_ROUNDS;
	uint32_t idx = (_idx & 0xFFFFFFFFULL) ^ (_idx >> 32);

	if (count < 8) {
		memset(result, 0, sizeof(size_t)*4);
		uint32_t ctr = 0;
		for (ssize_t i=0; i<count; ++i) {
			float xi  = sampleTEASingle(idx, ctr++, rounds),
				  cdf = 0.0f;

			int offset = 0;
			for (; offset<4; ++offset) {
				cdf = cdf + p[offset];
				if (xi < cdf)
					break;
			}

			result[std::min(offset, 3)]++;
		}
		return;
	}


	/* Compute a LDL^T decomposition of the (semidefinite) covariance matrix */
	MM_ALIGN16 float L[4][4];
	MM_ALIGN16 float D[4];
	memset(L, 0, sizeof(float)*4*4);
	memset(D, 0, sizeof(float)*4);

	float t = std::numeric_limits<float>::epsilon();
	for (int i=3; i>=0; --i) {
		D[i] = p[i] * t / (t + p[i]);
		L[i][i] = 1;
		for (int j=i+1; j<4; ++j)
			L[j][i] = -p[j]/t;
		t = t + p[i];
	}

	MM_ALIGN16 float result_f[4];
	#if USE_VECTORIZED == 1
		sampleGaussian4DVectorized(idx, D, (const float *) L, p, count, result_f);
	#else
		sampleGaussian4D(idx, D, L, p, count, result_f);
	#endif

	ssize_t actualCount = 0;
	for (int i=0; i<4; ++i) {
		size_t truncated = (size_t) std::max((ssize_t) 0, (ssize_t) ::round(result_f[i]));
		result[i] = truncated;
		actualCount += truncated;
	}

	if (count < 1000) {
		/* Rounding */
		uint32_t ctr = 4;
		while (actualCount != count) {
			float xi     = sampleTEASingle(idx, ctr++, rounds),
				  cdf    = 0.0f;

			int offset = 0;
			for (; offset<4; ++offset) {
				cdf = cdf + p[offset];
				if (xi < cdf)
					break;
			}

			offset = std::min(offset, 3);

			if (actualCount < count) {
				result[offset]++;
				actualCount++;
			} else if (result[offset] > 0) {
				result[offset]--;
				actualCount--;
			}
		}
	}
}

void sampleUniformMultinomialApprox(size_t _idx, ssize_t count, size_t *result) {
	const int rounds = RNG_ROUNDS;
	uint32_t idx = (_idx & 0xFFFFFFFFULL) ^ (_idx >> 32);

	if (count < 8) {
		memset(result, 0, sizeof(size_t)*4);
		uint32_t ctr = 0;
		for (ssize_t i=0; i<count; ++i) {
			int offset = std::min(3, (int) (sampleTEASingle(idx, ctr++, rounds) * 4));
			result[offset]++;
		}
		return;
	}

	/* LDL^T decomposition */
	MM_ALIGN16 const float L[4][4] = {
		{    1.f,  0.f,  0.f, 0.f },
		{ -1/3.f,  1.f,  0.f, 0.f },
		{ -1/3.f, -.5f,  1.f, 0.f },
		{ -1/3.f, -.5f, -1.f, 1.f }
	};

	MM_ALIGN16 const float D[4] = { 0.1875f, 1.0f/6.0f, 0.125f, 0.0f };
	MM_ALIGN16 const float p[4] = { 0.25f, 0.25f, 0.25f, 0.25f };

	MM_ALIGN16 float result_f[4];
	#if USE_VECTORIZED == 1
		sampleGaussian4DVectorized(idx, D, (const float *) L, p, count, result_f);
	#else
		sampleGaussian4D(idx, D, L, p, count, result_f);
	#endif

	ssize_t actualCount = 0;
	for (int i=0; i<4; ++i) {
		size_t truncated = (size_t) std::max((ssize_t) 0, (ssize_t) ::round(result_f[i]));
		result[i] = truncated;
		actualCount += truncated;
	}

	/* Rounding */
	if (count < 1000) {
		uint32_t ctr = 4;
		while (actualCount != count) {
			int offset = std::min(3, (int) (sampleTEASingle(idx, ctr++, rounds) * 4));

			if (actualCount < count) {
				result[offset]++;
				actualCount++;
			} else if (result[offset] > 0) {
				result[offset]--;
				actualCount--;
			}
		}
	}
}
void sampleUniformMultinomialApprox2(size_t _idx, ssize_t count, size_t *result) {
	const int rounds = RNG_ROUNDS;
	uint32_t idx = (_idx & 0xFFFFFFFFULL) ^ (_idx >> 32);

	if (count < 8) {
		memset(result, 0, sizeof(size_t) * 4);
		uint32_t ctr = 0;
		for (ssize_t i = 0; i<count; ++i) {
			int offset = std::min(3, (int)(sampleTEASingle(idx, ctr++, rounds) * 4));
			result[offset]++;
		}
		return;
	}
#if 1
	result[0] = result[1] = result[2] = result[3] = count / 4;

	if (result[0] * 4 < count)
		result[0] += count - result[0] * 4;
	return;
#endif


	/* LDL^T decomposition */
	MM_ALIGN16 const float L[4][4] = {
		{ 1.f, 0.f, 0.f, 0.f },
		{ -1 / 3.f, 1.f, 0.f, 0.f },
		{ -1 / 3.f, -.5f, 1.f, 0.f },
		{ -1 / 3.f, -.5f, -1.f, 1.f }
	};

	MM_ALIGN16 const float D[4] = { 0.1875f, 1.0f / 6.0f, 0.125f, 0.0f };
	MM_ALIGN16 const float p[4] = { 0.25f, 0.25f, 0.25f, 0.25f };

	MM_ALIGN16 float result_f[4];
#if USE_VECTORIZED == 1
	sampleGaussian4DVectorized(idx, D, (const float *)L, p, count, result_f);
#else
	sampleGaussian4D(idx, D, L, p, count, result_f);
#endif

	ssize_t actualCount = 0;
	for (int i = 0; i<4; ++i) {
		size_t truncated = (size_t)std::max((ssize_t)0, (ssize_t) ::round(result_f[i]));
		result[i] = truncated;
		actualCount += truncated;
	}

	/* Rounding */
	if (count < 1000) {
		uint32_t ctr = 4;
		while (actualCount != count) {
			int offset = std::min(3, (int)(sampleTEASingle(idx, ctr++, rounds) * 4));

			if (actualCount < count) {
				result[offset]++;
				actualCount++;
			}
			else if (result[offset] > 0) {
				result[offset]--;
				actualCount--;
			}
		}
	}
}
Float gammaVariate(Float a, uint32_t index, uint32_t &offset) {
	Float d = a - (Float) (1.0 / 3.0);
	Float c = (Float) (1.0 / 3.0) / std::sqrt(d);
	Float z, v, u;

	if (a == 0)
		return 0.0f;

	while (1) {
		do {
			z = warp::intervalToStdNormal(sampleTEAFloat(index, offset++));
			v = z*c +1;
		} while (v <= 0);
		v = v*v*v;
		u = sampleTEAFloat(index, offset++);
		if (u < 1 - 0.0331 * z * z * z * z)
			break;
		if (std::log(u) < 0.5*z*z + d*(1-v+std::log(v)))
			break;
	}
	return d*v;
}

void dirichletVariate(const Float *alpha, Float *result, uint32_t index) {
	Float sum = 0;
	uint32_t offset = 0;
	for (int i=0; i<4; ++i) {
		result[i] = gammaVariate(alpha[i], index, offset);
		sum += result[i];
	}
	if (sum != 0)
		sum = 1/sum;
	for (int i=0; i<4; ++i)
		result[i] *= sum;
}

MTS_NAMESPACE_END

#endif /* __SREF_UTILS_H */
