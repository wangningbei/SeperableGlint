/*
    This file is part of Mitsuba, a physically based rendering system.
	
	This is the implementation of paper "Fast Global Illumination with 
	Discrete Stochastic Microfacets Using a Filterable Model" by 
	Beibei Wang, Lu Wang, and Nicolas Holzschuch.

    Copyright (c) 2007-2012 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/qmc.h>
#include <mitsuba/core/statistics.h>
#include <boost/tr1/unordered_map.hpp>
#include "utils.h"
#include <queue>
#include <boost/filesystem/fstream.hpp>
#include <mitsuba/core/fstream.h>
#include <thinks/ppm.hpp>

#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif

#define ALPHA_SCALE  2.0f
#define GAMMA_SCALE  2
#define UNIFORM_PROB 0

#define PI_DIVIDE_180 0.0174532922
#define MAXV 360
#define TWO_PI 6.283185307

MTS_NAMESPACE_BEGIN

static StatsCounter avgExpansions("Stochastic reflectance model", "Avg. expansions/query", EAverage);
static StatsCounter avgExpansionsPdf("Stochastic reflectance model", "Avg. expansions/pdf query", EAverage);
static StatsCounter avgParticles("Stochastic reflectance model", "Avg. particles/query", EAverage);
static StatsCounter numClampedUV0("Stochastic reflectance model", "Queries clamped in UV space (0)");
static StatsCounter numClampedUV1("Stochastic reflectance model", "Queries clamped in UV space (1)");
static StatsCounter numClampedDir("Stochastic reflectance model", "Queries clamped in direction space");
static StatsCounter percHyperbolic("Stochastic reflectance model", "Hyperbolic case occurred", EPercentage);

struct Node {
	size_t id, numParticles;
	int depth;
	AABB2 uv;
	SphericalTriangle dir;
	AABB2 dirUV;

	/// Default constructor: initialize with the entire domain
	inline Node(size_t numParticles)
		: id(1), numParticles(numParticles), depth(0),
		uv(Point2(0, 0), Point2(1, 1)), dirUV(Point2(0, 0), Point2(1, 1)), dir(1) {
	}

	inline Node() { }

	/// Check if the rectangle is empty (i.e. there are no particles)
	inline bool empty() const { return numParticles == 0; }

	/// Return a string representation
	inline std::string toString() const {
		std::ostringstream oss;
		oss << "Node[id=" << id << ", depth=" << depth << ", uv="
			<< uv.toString() << ", dir=" << dir.toString() << ", particles=" << numParticles << "]";
		return oss.str();
	}
};

struct SpaceNode {
	size_t id, numParticles;
	int depth;
	AABB2 uv;

	/// Default constructor: initialize with the entire domain
	inline SpaceNode(size_t numParticles)
		: id(1), numParticles(numParticles), depth(0),
		uv(Point2(0, 0), Point2(1, 1)) {
	}

	inline SpaceNode() { }

	/// Check if the rectangle is empty (i.e. there are no particles)
	inline bool empty() const { return numParticles == 0; }

	/// Return a string representation
	inline std::string toString() const {
		std::ostringstream oss;
		oss << "SpaceNode[id=" << id << ", depth=" << depth << ", uv="
			<< uv.toString() << ", particles=" << numParticles << "]";
		return oss.str();
	}
};

struct QueueNode : public Node {
	QueueNode(const Node &node, EIntersectionResult intersectionUV, EIntersectionResult intersectionDir)
		: Node(node), intersectionUV(intersectionUV), intersectionDir(intersectionDir) { }
	QueueNode(const Node &node, EIntersectionResult intersectionUV)
		: Node(node), intersectionUV(intersectionUV){ }
	QueueNode(size_t particleCount) : Node(particleCount),
		intersectionUV(EIntersection), intersectionDir(EIntersection) { }
	QueueNode(){}
	EIntersectionResult intersectionUV, intersectionDir;
};
struct QueueSpaceNode : public SpaceNode {
	QueueSpaceNode(const SpaceNode &node, EIntersectionResult intersectionUV)
		: SpaceNode(node), intersectionUV(intersectionUV){ }
	QueueSpaceNode(size_t particleCount) : SpaceNode(particleCount),
		intersectionUV(EIntersection) { }
	QueueSpaceNode(){}
	EIntersectionResult intersectionUV;
};

static PrimitiveThreadLocal<std::queue<QueueNode >> __queueTLS[40];
static PrimitiveThreadLocal<std::stack<QueueSpaceNode >> __stackTLS;


class PreStochasticReflectance : public BSDF {
public:
	PreStochasticReflectance(const Properties &props)
		: BSDF(props) {

		props.markQueried("distribution");
		m_particleCount = (size_t)(props.getSize("particleCount") * props.getFloat("particleMultiplier", 1.0f));
		m_eta = props.getFloat("eta", 0.0f);
		m_wiGamma = props.getFloat("gamma", 1.0f);// 
		m_gamma = degToRad(m_wiGamma);
		Float alpha = props.getFloat("alpha", 1.33f);
		m_alphaU = props.getFloat("alphaU", alpha);
		m_alphaV = props.getFloat("alphaV", alpha);
		m_avgQueryArea = props.getFloat("queryArea", 1e-5f);
		m_reflectance = props.getSpectrum("specularReflectance", props.getSpectrum("reflectance", Spectrum(1.0f)));
		m_diffuseReflectance = props.getSpectrum("diffuseReflectance", Spectrum(0.0f));
		m_spatialLookupMultiplier = props.getFloat("spatialLookupMultiplier", 1);
		m_maxDepth = props.getInteger("maxDepth", 20);
		m_errorThreshold = props.getFloat("errorThreshold", 0.1);
		m_collectStatistics = props.getBoolean("collectStatistics", false);
		m_clamp = props.getBoolean("clamp", true);
		m_statUVArea = 0;
		
		m_cosGamma = std::cos(m_gamma);
		m_avgQuerySolidAngle = .5f*M_PI*(1 - m_cosGamma);
		m_statQueryArea = 0;
		m_statQueries = 0;

		Log(EInfo, "Precomputing triangle integrals ..");
		ref<Timer> timer = new Timer();
		Vector4f integrals;
		m_triIntegrals.clear();
		for (int i = 2; i < 6; ++i)
			integrals[i - 2] = precomputeIntegrals(SphericalTriangle(i));
		Float total = integrals[0] + integrals[1] + integrals[2] + integrals[3];
		m_triIntegrals[1] = integrals / total;
		Log(EInfo, "Done. (took %i ms, %i entries, "
			"normalization = %f)", timer->getMilliseconds(),
			(int)m_triIntegrals.size(), total);

		m_depthfac = 0.55; //this is the factor of the spatial depth, 0.55 is a set for all of our test scenes.

		m_space_p = props.getBoolean("spaceProb", false); //use the filterable model
		m_particle_thresh = props.getFloat("spaceProbThresh", 16);

		setupPrecomputedTable();

	}

	void setupPrecomputedTable()
	{
		m_dir_step = 1.0;
		m_res_u = 90 / m_dir_step;
		m_res_v = MAXV / m_dir_step;
		m_res_u_min_1 = m_res_u - 1;
		m_res_v_min_1 = m_res_v - 1;

		const int blurAngle[levelCount] = { 90, 60, 45, 30, 20, 10, 5, 2, 1 };
		for (int i = 0; i < levelCount; i++)
		{
			level[i] = blurAngle[i];
		}
		m_inv_dir_step = 57.2957804904 / m_dir_step;

		SLog(EDebug, "The directional step is %f", m_dir_step);
	
		m_particleCountTableMipmap.resize(levelCount);
		for (int ilevel = 0; ilevel < levelCount; ilevel++)
		{
			m_stepDirMipMap[ilevel] = level[ilevel];
			m_res_u_MipMap[ilevel] = 90 / m_stepDirMipMap[ilevel];
			m_res_v_MipMap[ilevel] = MAXV / m_stepDirMipMap[ilevel];

			m_res_u_min_1_MipMap[ilevel] = m_res_u_MipMap[ilevel] - 1;
			m_res_v_min_1_MipMap[ilevel] = m_res_v_MipMap[ilevel] - 1;

			m_inv_dir_step_MipMap[ilevel] = 57.2957804904 / m_stepDirMipMap[ilevel];

			int temp_u = m_res_u_MipMap[ilevel];
			int temp_v = m_res_v_MipMap[ilevel];
			m_particleCountTableMipmap[ilevel].resize(temp_u);

			for (int i = 0; i < temp_u; i++)
			{
				m_particleCountTableMipmap[ilevel][i].resize(temp_u);

				for (int j = 0; j < temp_u; j++)
				{
					m_particleCountTableMipmap[ilevel][i][j].resize(temp_v, 0);
				}
			}
		}

		std::vector<std::vector<std::vector<float>>> particleCountTable;
		std::vector<std::vector<std::vector<float>>> particleCountTableWi;
		particleCountTable.resize(m_res_u);
		particleCountTableWi.resize(m_res_u);
		for (int i = 0; i < m_res_u; i++)
		{
			particleCountTable[i].resize(m_res_u);
			particleCountTableWi[i].resize(m_res_u);
			for (int j = 0; j < m_res_u; j++)
			{
				particleCountTable[i][j].resize(m_res_v, 0);
				particleCountTableWi[i][j].resize(m_res_v, 0);
			}
		}

		Log(EInfo, "Start the precompuation of the directional partical count.");
		precomputeDirParticleCountTable(particleCountTable);
		Log(EInfo, "Start the gamma accumulation.");
		precomputeDirParticleCountTableGamma(particleCountTable, particleCountTableWi);
		Log(EInfo, "Start the Filter.");
		createMipmapWithGaussainFilter(particleCountTableWi);
		Log(EInfo, "Done.");
	}

	PreStochasticReflectance(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_particleCount = stream->readSize();
		m_alphaU = stream->readFloat();
		m_alphaV = stream->readFloat();
		m_eta = stream->readFloat();
		m_wiGamma = stream->readFloat();
		m_gamma = stream->readFloat();
		m_avgQueryArea = stream->readFloat();
		m_reflectance = Spectrum(stream);
		m_diffuseReflectance = Spectrum(stream);
		m_spatialLookupMultiplier = stream->readFloat();
		m_maxDepth = stream->readInt();
		m_errorThreshold = stream->readFloat();
		m_collectStatistics = stream->readBool();
		m_clamp = stream->readBool();
		m_statUVArea = 0;

		m_depthfac = 0.55;
		m_space_p = stream->readBool();
		m_particle_thresh = stream->readFloat();
		setupPrecomputedTable();
		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
		stream->writeSize(m_particleCount);
		stream->writeFloat(m_alphaU);
		stream->writeFloat(m_alphaV);
		stream->writeFloat(m_eta);
		stream->writeFloat(m_wiGamma);
		stream->writeFloat(m_gamma);		
		stream->writeFloat(m_avgQueryArea);
		m_reflectance.serialize(stream);
		m_diffuseReflectance.serialize(stream);
		stream->writeFloat(m_spatialLookupMultiplier);
		stream->writeInt(m_maxDepth);
		stream->writeFloat(m_errorThreshold);
		stream->writeBool(m_collectStatistics);
		stream->writeBool(m_clamp);

		stream->writeBool(m_space_p);
		stream->writeFloat(m_particle_thresh);
	}

	void configure() {
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide);
		m_usesRayDifferentials = true;
		BSDF::configure();

		//set the level of the mipmap
		float peakValue = microfacetD(Vector(0, 0, 1));
		//the difference with the peakValue could be 0.8 of it.
		float allowedValue = peakValue * 0.8;
		float degree = 0;
		float tempValue = peakValue;
		for (int i = 1; i < 90; i++)
		{
			float theta1 = i * PI_DIVIDE_180;
			float cosTheta1 = cos(theta1);
			Vector halfV = Vector(sqrt(1 - cosTheta1 * cosTheta1), 0, cosTheta1);
			tempValue = microfacetD(halfV);
			if (tempValue <= allowedValue)
			{
				degree = i;
				break;
			}
		}
		if (tempValue > allowedValue)
			degree = 90;

		for (int i = 0; i < levelCount - 1; i++)
		{
			if (degree <= level[i] && degree >= level[i + 1])
				m_i_level = i + 1;
		}

		Log(EInfo, "The selected level is %i with blur angle %i.", m_i_level, level[m_i_level]);

		if (m_i_level != 8)
		{
			m_inv_dir_step = m_inv_dir_step_MipMap[m_i_level];
			m_dir_step = m_stepDirMipMap[m_i_level];

			m_res_u = 90 / m_dir_step;
			m_res_v = MAXV / m_dir_step;

			m_res_u_min_1 = m_res_u - 1;
			m_res_v_min_1 = m_res_v - 1;
		}
	}

	//From Jakob 2014
	Float precomputeIntegrals(const SphericalTriangle &tri) {
		Float rule1 = microfacetD(normalize(tri.v0 + tri.v1 + tri.v2)),
			rule2 = (microfacetD(tri.v0) + microfacetD(tri.v1) + microfacetD(tri.v2)) * (1.0f / 3.0f),
			error = std::abs(rule1 - rule2),
			area = tri.area();

		Float integral = 0;
		if (error * area < 1e-4f || error < 1e-4f * rule2) {
			integral = rule2 * area;
		}
		else {
			SphericalTriangle children[4];
			tri.split(children);

			Vector4f recursiveIntegrals;
			for (int i = 0; i<4; ++i) {
				Float nested = precomputeIntegrals(children[i]);
				recursiveIntegrals[i] = nested;
				integral += nested;
			}
			if (integral != 0)
				m_triIntegrals[tri.id] = recursiveIntegrals / integral;
			else
				m_triIntegrals[tri.id] = Vector4f(0.25f);
		}
		return integral;
	}

	inline void splitSpaceFake(const Node &n, Node *ch) const {
		int i = 3;
		ch[i].id = 4 * n.id + i - 2;
		ch[i].dir = n.dir;
		ch[i].numParticles = n.numParticles;
		ch[i].depth = n.depth + 1;
		ch[i].uv = n.uv.getChild(i);
	}

	inline void splitDirection(const Node &n, Node *ch) const {
		n.dir.split(ch[0].dir, ch[1].dir, ch[2].dir, ch[3].dir);

		float binProbs[4];
		std::tr1::unordered_map<size_t, Vector4f>::const_iterator it = m_triIntegrals.find(n.dir.id);
		if (it != m_triIntegrals.end()) {
			const Vector4f &probs = it->second;
			for (int i = 0; i<4; ++i)
				binProbs[i] = probs[i];
		}
		else
		{
			for (int i = 0; i<4; ++i)
				binProbs[i] = (float)(microfacetD(ch[i].dir.center()) * ch[i].dir.area());
			float normalization = 1 / (binProbs[0] + binProbs[1] + binProbs[2] + binProbs[3]);
			for (int i = 0; i<4; ++i)
				binProbs[i] *= normalization;
		}

		size_t binCounts[4];
		sampleMultinomialApprox(n.id, binProbs, n.numParticles, binCounts);
		for (int i = 0; i < 4; ++i) {
			ch[i].id = 4 * n.id + i - 2;
			ch[i].numParticles = binCounts[i];
			ch[i].depth = n.depth + 1;
			ch[i].uv = n.uv;
		}
	}

	//pre compute the directional domain particle count
	Float countParticlesForDir(QueryRegion &query, float t_gamma, std::queue<QueueNode> &queue) const {

		float t_avgQuerySolidAngle = .5f*M_PI*(1 - t_gamma);
		const int t_particleCount = 1e9;

		queue.push(QueueNode(t_particleCount));
		const int directionalDepth = m_maxDepth;
		const int Ni = t_particleCount;

		Float result = 0;
		while (!queue.empty()) {
			query.nExpansions++;
			QueueNode node = queue.front();
			queue.pop();
			if ((result > 0 || node.depth > directionalDepth) && node.dir.area() < 0.5 && query.dir.ellipse  && m_triIntegrals.find(node.dir.id) == m_triIntegrals.end()) // //
			{
				Float overlap = query.dir.sphTriangleOverlap(node.dir);
				Float expError2 = std::sqrt(node.numParticles * overlap * (1 - overlap));
				Float threshold = m_errorThreshold * result;

				if (expError2 <= threshold * threshold || node.depth > directionalDepth) // 
				{
					result += node.numParticles *overlap;
					continue;
				}
			}

			Node children[4];
			bool subdivideSpace = t_avgQuerySolidAngle * node.uv.getVolume() > m_avgQueryArea * node.dir.area();
			if (subdivideSpace) {
				splitSpaceFake(node, children);
				queue.push(QueueNode(children[3], node.intersectionUV, node.intersectionDir));
			}
			else
			{
				splitDirection(node, children);
				EIntersectionResult intersectionDir = node.intersectionDir;
				for (int i = 0; i < 4; ++i) {
					if (children[i].empty())
						continue;
					if (node.intersectionDir != EInside) {
#if defined(SINGLE_PRECISION)
						intersectionDir = query.dir.intersectSphTriangleVectorized(children[i].dir);
#else
						intersectionDir = query.dir.intersectSphTriangle(children[i].dir);
#endif
						if (intersectionDir == EInside)
						{ 
							result += children[i].numParticles;
							continue;
						}
					}
					if (intersectionDir != EDisjoint)
						queue.push(QueueNode(children[i], node.intersectionUV, intersectionDir));
				}
			}
		}

		return (float)result / (float)Ni;
	}


	float precomputeForDirection(const Vector &wi, const Vector &wo, std::queue<QueueNode> &queue)
	{
		Parallelogram2 dumpuv;//this is dump one
		dumpuv.o = Point2(0.5, 0.5);
		dumpuv.v0 = Vector2(0, 1);
		dumpuv.v1 = Vector2(1, 0);
		dumpuv.computeBoundingBox();
		float temp_gamma = degToRad(1.0);
		QueryRegion query(dumpuv, SphericalConic(wi, wo, temp_gamma, m_clamp));

		return countParticlesForDir(query, temp_gamma, queue);
	}

	void precomputeDirParticleCountTable(std::vector<std::vector<std::vector<float>>>& particleCountTable)
	{
		float offset = 0.5f;

		const int threadCount = mts_omp_get_max_threads();
		std::vector<std::queue<QueueNode >> queues(threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			queues[i] = __queueTLS[i].get();
		}

#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif	
		for (int i = 0; i < m_res_u; i++)
		{
#if defined(MTS_OPENMP)
			int tid = mts_omp_get_thread_num();
#else
			int tid = 0;
#endif
			float theta1 = (i + offset)* m_dir_step * PI_DIVIDE_180;
			float cosTheta1 = cos(theta1);
			Vector wo = Vector(sqrt(1 - cosTheta1 * cosTheta1), 0, cosTheta1);

			for (int j = 0; j < m_res_u; j++)
			{
				float theta2 = (j + offset)* m_dir_step * PI_DIVIDE_180;
				float cosTheta2 = cos(theta2);

				for (int k = 0; k < m_res_v; k++)
				{
					float dPhi = (k + offset)* m_dir_step * PI_DIVIDE_180;
					float sinTheta2 = sqrt(1 - cosTheta2 * cosTheta2);
					Vector wi = Vector(sinTheta2* cos(dPhi), sinTheta2 * sin(dPhi), cosTheta2);
					if (wi == wo) continue;
					particleCountTable[i][j][k] = precomputeForDirection(wi, wo,queues[tid]);
				}
			}	
		}
	
	}

	void precomputeDirParticleCountTableGamma(const std::vector<std::vector<std::vector<float>>> &particleCountTable,		
		std::vector<std::vector<std::vector<float>>> &particleCountTableWi)
	{

		int temp_gamma_0 = m_wiGamma * 0.5, temp_gamma1 = m_wiGamma * 0.5;
		if (m_wiGamma == 1) { temp_gamma_0 = 0;  temp_gamma1 = 1; }
		const float offset = 0.5;

#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
		for (int i = 0; i < m_res_u; i++)
		{
			for (int j = 0; j < m_res_u; j++)
			{
				int theta2 = (j + offset) * m_dir_step;
				for (int k = 0; k < m_res_v; k++)
				{
					for (float wii = -temp_gamma_0; wii < temp_gamma1; wii += 1)
					{
						for (float wij = -temp_gamma_0; wij < temp_gamma1; wij += 1)
						{
							//we need a interpolation here
							int theta1 = (i + offset) * m_dir_step + wii;
							int phi0 = (k + offset)* m_dir_step + wij;
							if (phi0 < 0) phi0 += m_res_v;
							if (phi0 >= m_res_v) phi0 -= m_res_v;
							if (theta1 < 0 || theta1 >= m_res_u) continue;

							particleCountTableWi[i][j][k] += particleCountTable[theta1][theta2][phi0];
						}
					}
				}
			}
		}
	}

	inline float gaussian3D(const Vector dist, const float D)
	{
		float c1 = M_PI * 2 * D * D;
		float c2 = 1.0 / pow(c1, 1.0);
		float c3 = -(dist.lengthSquared() / (2 * D * D));
		return c2 * exp(c3);	
	}

	void createMipmapWithGaussainFilter(const std::vector<std::vector<std::vector<float>>> &particleCountTableWi)
	{
		for (int ilevel = 0; ilevel < levelCount; ilevel++)
		{
			int t_dir_step = m_stepDirMipMap[ilevel];
			int t_res_u = m_res_u_MipMap[ilevel];
			int t_res_v = m_res_v_MipMap[ilevel];

#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
			for (int i = 0; i < t_res_u; i++)
			{
				for (int j = 0; j < t_res_u; j++)
				{
					for (int k = 0; k < t_res_v; k++)
					{
						float facSum = 0;
						for (int wii = 0; wii < t_dir_step; wii += 1)
						{
							for (int wij = 0; wij < t_dir_step; wij += 1)
							{
								for (int phj = 0; phj < t_dir_step; phj += 1)
								{
									int theta10 = i * t_dir_step + wii;
									int theta20 = j * t_dir_step + wij;
									int phi0 = k * t_dir_step + phj;									

									float factor = gaussian3D(Vector(theta10, theta20, phi0) + Vector(0.5) 
										- Vector(i + 0.5, j + 0.5, k + 0.5) * t_dir_step,
										t_dir_step * 0.5);
									facSum += factor;

									m_particleCountTableMipmap[ilevel][i][j][k] += factor * particleCountTableWi[theta10][theta20][phi0];
								}
							}
						}
						m_particleCountTableMipmap[ilevel][i][j][k] /= facSum;
					}
				}
			}
		}
	}


	inline Float microfacetD(const Vector &v, bool sample=false) const {
		Float result = 0;
		Float alphaU = m_alphaU, alphaV = m_alphaV;
		if (sample) {
			alphaU *= ALPHA_SCALE; alphaV *= ALPHA_SCALE;
		}
		if (EXPECT_TAKEN(alphaU == alphaV)) {
			Float mu = v.z, mu2 = mu * mu, mu3 = mu2 * mu;

			if (mu == 0)
				return 0;

			result = std::exp((mu2-1)/(mu2*alphaU*alphaU)) / (M_PI*mu3*alphaU*alphaU);
		} else {
			alphaU = std::max(2 / (alphaU * alphaU) - 2, (Float) 0.1f);
			alphaV = std::max(2 / (alphaV * alphaV) - 2, (Float) 0.1f);

			const Float mu = v.z;
			const Float ds = 1 - mu * mu;
			if (ds < 0)
				return 0.0f;
			const Float exponent = (alphaU * v.x * v.x
					+ alphaV * v.y * v.y) / ds;
			result = std::sqrt((alphaU + 2) * (alphaV + 2))
				* INV_TWOPI * std::pow(mu, exponent);
		}
		return result;
	}

	inline Vector sampleMicrofacetD(const Point2 &sample) const {
		Float cosThetaM, phiM;

		Float alphaU = m_alphaU, alphaV = m_alphaV;
		alphaU *= ALPHA_SCALE; alphaV *= ALPHA_SCALE;

		if (EXPECT_TAKEN(alphaU == alphaV)) {
			Float tanThetaMSqr = -alphaU*alphaU* math::fastlog(1.0f - sample.x);
			cosThetaM = 1.0f / std::sqrt(1 + tanThetaMSqr);
			phiM = (2.0f * M_PI) * sample.y;
		} else {
			alphaU = std::max(2 / (alphaU * alphaU) - 2, (Float) 0.1f);
			alphaV = std::max(2 / (alphaV * alphaV) - 2, (Float) 0.1f);

			/* Sampling method based on code from PBRT */
			if (sample.x < 0.25f) {
				sampleFirstQuadrantAS(alphaU, alphaV,
					4 * sample.x, sample.y, phiM, cosThetaM);
			} else if (sample.x < 0.5f) {
				sampleFirstQuadrantAS(alphaU, alphaV,
					4 * (0.5f - sample.x), sample.y, phiM, cosThetaM);
				phiM = M_PI - phiM;
			} else if (sample.x < 0.75f) {
				sampleFirstQuadrantAS(alphaU, alphaV,
					4 * (sample.x - 0.5f), sample.y, phiM, cosThetaM);
				phiM += M_PI;
			} else {
				sampleFirstQuadrantAS(alphaU, alphaV,
					4 * (1 - sample.x), sample.y, phiM, cosThetaM);
				phiM = 2 * M_PI - phiM;
			}
		}

		const Float sinThetaM = std::sqrt(
			std::max((Float) 0, 1 - cosThetaM*cosThetaM));

		Float sinPhiM, cosPhiM;
		math::sincos(phiM, &sinPhiM, &cosPhiM);

		return Vector(
			sinThetaM * cosPhiM,
			sinThetaM * sinPhiM,
			cosThetaM
		);

	}
	/// Helper routine: sample the first quadrant of the A&S distribution
	inline void sampleFirstQuadrantAS(Float alphaU, Float alphaV, Float u1, Float u2,
			Float &phi, Float &cosTheta) const {
		if (alphaU == alphaV)
			phi = M_PI * u1 * 0.5f;
		else
			phi = std::atan(
				std::sqrt((alphaU + 1.0f) / (alphaV + 1.0f)) *
				std::tan(M_PI * u1 * 0.5f));
		const Float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
		cosTheta = std::pow(u2, 1.0f /
			(alphaU * cosPhi * cosPhi + alphaV * sinPhi * sinPhi + 1.0f));
	}

	Float smithG1(const Vector &v, const Vector &m) const {
		Float alpha;
		if (EXPECT_TAKEN(m_alphaU == m_alphaV)) {
			alpha = m_alphaU;
		} else {
			alpha = std::max(m_alphaU, m_alphaV);
		}

		const Float tanTheta = std::abs(Frame::tanTheta(v));

		/* perpendicular incidence -- no shadowing/masking */
		if (tanTheta == 0.0f)
			return 1.0f;

		/* Can't see the back side from the front and vice versa */
		if (dot(v, m) * Frame::cosTheta(v) <= 0)
			return 0.0f;

		Float a = 1.0f / (alpha * tanTheta);

		if (a >= 1.6f)
			return 1.0f;

		/* Use a fast and accurate (<0.35% rel. error) rational
		   approximation to the shadowing-masking function */
		const Float aSqr = a * a;
		return (3.535f * a + 2.181f * aSqr)
			 / (1.0f + 2.276f * a + 2.577f * aSqr);
	}


	inline void splitSpaceOnly(const SpaceNode &n, SpaceNode *ch) const {
		size_t binCounts[4];
		sampleUniformMultinomialApprox(n.id, n.numParticles, binCounts);

		for (int i = 0; i<4; ++i) {
			ch[i].id = 4 * n.id + i - 2;
			ch[i].numParticles = binCounts[i];
			ch[i].depth = n.depth + 1;
			ch[i].uv = n.uv.getChild(i);
		}
	}

	// count the particles which locate in the query footprint
	Float countParticlesSpatial(QueryRegion &query, const float &pd, const Vector3i &indexDir,
		const int &depth) const {

		std::stack<QueueSpaceNode> &queue = __stackTLS.get();
		queue.push(QueueSpaceNode(m_particleCount));
		const int maxDepthSpa = m_maxDepth * m_depthfac;

		int idDir = (depth < 0) ? 0 : indexDir.x * m_res_u * m_res_v + indexDir.y * m_res_v;// +indexDir.z;
		Float result = 0;
		float uvarea = abs(det(Vector2(query.uv.bbox.min), Vector2(query.uv.bbox.max)));

		if (uvarea == 0) return 0;
		while (!queue.empty()) {
			QueueSpaceNode node = queue.top();
			queue.pop();

			if (node.depth > maxDepthSpa) {
				Float overlap = std::abs(query.uv.overlapAABB(node.uv)) / node.uv.getVolume();
				if (overlap > 0)
				{
					int index = node.id + idDir;
					uint32_t idx = (index & 0xFFFFFFFFULL) ^ (index >> 32);
					for (int k = 0; k < node.numParticles; k++)
					{
						float xi = sampleTEASingle(idx, k, 8);
						result += overlap *(xi < pd); 
					}					
				}
				continue;
			}

			SpaceNode children[4];
			{
				splitSpaceOnly(node, children);
				EIntersectionResult intersectionUV;
				for (int i = 0; i < 4; ++i) {
					if (children[i].empty())
						continue;
					intersectionUV = query.uv.intersectAABB(children[i].uv);
					if (intersectionUV == EInside)
					{
						int index = children[i].id + idDir;
						uint32_t idx = (index & 0xFFFFFFFFULL) ^ (index >> 32);
						for (int k = 0; k < children[i].numParticles; k++)
						{
							float xi = sampleTEASingle(idx, k, 8);
							result += (xi < pd);
						}
						continue;
					}

					if (intersectionUV != EDisjoint)
						queue.push(QueueSpaceNode(children[i], intersectionUV));
				}
			}
		}

		return result;
	}

	//read the DPF from the precomputed table
	float readDPF(const BSDFSamplingRecord &bRec, Vector3i &dirIdx) const
	{
		//we reuse the its.time for the path depth, not cool
		const int pathDepth = (bRec.its.time);

		float inv_dir_step = m_inv_dir_step;
		int res_u_min_1 = m_res_u_min_1;
		int res_v_min_1 = m_res_v_min_1;
		int ilevel = m_i_level;

		float theta1 = acos(bRec.wo.z) * inv_dir_step;
		float theta2 = acos(bRec.wi.z) * inv_dir_step;
		float wiphi = atan2(bRec.wi.y, bRec.wi.x);
		if (wiphi < 0) wiphi += TWO_PI;

		float wophi = atan2(bRec.wo.y, bRec.wo.x);
		if (wophi < 0) wophi += TWO_PI;
		int wpphi0 = wophi * inv_dir_step;
		float dPhi = abs(wiphi - wophi)* inv_dir_step;

		int theta10 = std::max(0, std::min(math::roundToInt(theta1), res_u_min_1));
		int theta20 = std::max(0, std::min(math::roundToInt(theta2), res_u_min_1));
		int phi0 = std::max(0, std::min(math::roundToInt(dPhi), res_v_min_1));
		float pd = m_particleCountTableMipmap[ilevel][theta10][theta20][phi0];
		dirIdx = Vector3i(theta10, wpphi0, phi0);

		return pd;
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EGlossyReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		Float D;
		Vector H = normalize(bRec.wi + bRec.wo);

		if (bRec.its.hasUVPartials) {
			Vector2 v0(bRec.its.dudx, bRec.its.dvdx),
				v1(bRec.its.dudy, bRec.its.dvdy);

			v0 *= m_spatialLookupMultiplier;
			v1 *= m_spatialLookupMultiplier;

			Parallelogram2 uv(bRec.its.uv, v0, v1);
			bool clamped = false;

			QueryRegion query(uv, SphericalConic(bRec.wi, bRec.wo, m_gamma, m_clamp, &clamped));
			if (clamped)
				++numClampedDir;

			percHyperbolic.incrementBase();
			if (!query.dir.ellipse)
				++percHyperbolic;

			query.uv.computeBoundingBox();
			const double area = query.uv.area();
			if (area <= 0)
				return Spectrum(0.0);

			Float overlap = std::abs(query.uv.overlapAABB(
				AABB2(Point2(0, 0), Point2(1, 1))));

			if (overlap <= 0)
				return Spectrum(0.0);

			const int pathDepth = (bRec.its.time);//Fix this later.
			Vector3i dirIdx;
			float pd = readDPF(bRec, dirIdx);
			double nParticles = 0;

			//use the filterable model
			double nParticlesProb = (overlap * m_particleCount * pd);
			if (m_space_p && (nParticlesProb > m_particle_thresh))
			{
				nParticles = pd < 1e-4 ? 0 : nParticlesProb;
			}
			else
			{
				nParticles = pd < 1e-4 ? 0 : (countParticlesSpatial(query, pd, dirIdx, pathDepth));
			}

			D = (nParticles == 0 ? 0 : nParticles) / (area * 2 * M_PI*(1.0f - m_cosGamma) * m_particleCount);

			if (m_collectStatistics) {
				avgParticles += (size_t)round(nParticles);
				avgParticles.incrementBase(1);
				avgExpansions += query.nExpansions;
				avgExpansions.incrementBase(1);
				Float queryArea = atomicAdd(&m_statQueryArea, area);
				int64_t numQueries = atomicAdd(&m_statQueries, 1);
				if (numQueries == 1000000) {
					cout << toString() << ": average query area=" << queryArea / numQueries
						<< endl;
				}
			}
		}
		else
		{
			D = microfacetD(H) / (4 * absDot(H, bRec.wo));
		}
		//From Jakob 2014
		Float G = smithG1(bRec.wi, H) * smithG1(bRec.wo, H);
		Float F = m_eta != 0 ? fresnelDielectricExt(dot(bRec.wi, H), m_eta) : (Float)1;

		/* Calculate the specular reflection component */
		Float value = D * F * G * absDot(H, bRec.wo) / (Frame::cosTheta(bRec.wi));

		return m_reflectance * value;

	}

	//The pdf is the same as the Jakob 2014 paper
	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EGlossyReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		Float result = 0;
		if (bRec.its.hasUVPartials) {
				Frame frame(bRec.wo);
				int nSamples = 64;
				#if GAMMA_SCALE == 1
					Float cosGamma = m_cosGamma;
				#else
					Float cosGamma = std::cos(m_gamma * GAMMA_SCALE);
				#endif
				for (int i=0; i<nSamples; ++i) {
					Vector perturb = warp::squareToUniformCone(cosGamma, sample02(i));
					Vector wo = frame.toWorld(perturb);
					if (wo.z <= 0)
						continue;
					Vector H = normalize(bRec.wi + wo);
					/* Jacobian of the half-direction mapping */
					const Float dwh_dwo = 1.0f / (4.0f * dot(wo, H));
					result += microfacetD(H, true) * dwh_dwo;
				}
				result *= 1.0f / nSamples;
		} else {
			Vector H = normalize(bRec.wi + bRec.wo);
			/* Jacobian of the half-direction mapping */
			const Float dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));

			result = microfacetD(H) * dwh_dwo;
		}

		#if UNIFORM_PROB == 0
			return result;
		#else
			return result * (1-UNIFORM_PROB) + UNIFORM_PROB * INV_TWOPI;
		#endif
	}

	//The sampling is the same as the Jakob 2014 paper
	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		Point2 sample(sample_);
		if (EXPECT_TAKEN(sample.x >= UNIFORM_PROB)) {
			sample.x = (sample.x - UNIFORM_PROB) * 1.0f/(1.0f-UNIFORM_PROB);

			Vector m = sampleMicrofacetD(sample);

			bRec.wo = reflect(bRec.wi, m);

			if (bRec.its.hasUVPartials) {
				Point2 sample2 = bRec.sampler->next2D();
				#if GAMMA_SCALE == 1
					Float cosGamma = m_cosGamma;
				#else
					Float cosGamma = std::cos(m_gamma * GAMMA_SCALE);
				#endif
				Vector perturb = warp::squareToUniformCone(cosGamma, sample2);
				bRec.wo = Frame(bRec.wo).toWorld(perturb);
			}

			if (Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);
		} else {
			bRec.wo = warp::squareToUniformHemisphere(sample);
		}

		pdf_ = pdf(bRec, ESolidAngle);
		Spectrum value = eval(bRec, ESolidAngle);
		if (pdf_ < RCPOVERFLOW || value.isZero())
			return Spectrum(0.0f);
		return value / pdf_; 
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return PreStochasticReflectance::sample(bRec, pdf, sample);
	}

	void setParent(ConfigurableObject *obj) {
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		BSDF::addChild(name, child);
	}

	/* Unsupported / unimplemented operations */
	Float getRoughness(const Intersection &its, int component) const { return std::numeric_limits<Float>::infinity(); }
	Spectrum getDiffuseReflectance(const Intersection &its) const { return Spectrum(0.0f); }
	Float getRoughness() const {
		return m_alphaU;
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "PreStochasticReflectance[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  particleCount = " << m_particleCount << "," << endl
			<< "  uvArea = " << m_statUVArea << "," << endl
			<< "  actualParticleCount = \"" << m_particleCount * m_statUVArea << "," << endl
			<< "  alphaU = " << m_alphaU << "," << endl
			<< "  alphaV = " << m_alphaV << "," << endl
			<< "  gamma = " << m_gamma << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	// from Jakob 2014 paper
	std::tr1::unordered_map<size_t, Vector4f> m_triIntegrals;
	size_t m_particleCount;
	Float m_alphaU, m_alphaV;
	Float m_gamma, m_cosGamma;
	Float m_eta;
	Float m_avgQuerySolidAngle;
	Float m_avgQueryArea;
	Spectrum m_reflectance;
	Spectrum m_diffuseReflectance;
	Float m_spatialLookupMultiplier;
	int m_maxDepth;
	Float m_errorThreshold;
	bool m_collectStatistics;
	bool m_clamp;
	mutable Float m_statQueryArea;
	mutable int64_t m_statQueries;
	Float m_statUVArea;
	/********************************/

	//for directional domain particle count storage
	std::vector<std::vector<std::vector<std::vector<float>>>> m_particleCountTableMipmap;
	int m_res_u, m_res_v, m_res_u_min_1, m_res_v_min_1;
	
	float m_dir_step, m_inv_dir_step;
	float m_wiGamma, m_depthfac;

	//for mipmap
	const static int levelCount = 9;
	int level[levelCount];
	float m_stepDirMipMap[levelCount];
	int m_res_u_MipMap[levelCount];
	int m_res_v_MipMap[levelCount];
	int m_res_u_min_1_MipMap[levelCount];
	int m_res_v_min_1_MipMap[levelCount];
	float m_inv_dir_step_MipMap[levelCount];
	int m_i_level;

	//for filterable model
	bool m_space_p;
	float m_particle_thresh;
};


/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class PreStochasticShader : public Shader {
public:
	//)
	PreStochasticShader(Renderer *renderer, const Texture *specularReflectance,
		const Texture *diffuseReflectance,
		const Texture *alphaU, const Texture *alphaV)
		: Shader(renderer, EBSDFShader),
		m_specularReflectance(specularReflectance), m_diffuseReflectance(diffuseReflectance),
		m_alphaU(alphaU), m_alphaV(alphaV){
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
		m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());
		m_diffuseReflectanceShader = renderer->registerShaderForResource(m_diffuseReflectance.get());
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			m_diffuseReflectanceShader.get() != NULL &&
			m_alphaUShader.get() != NULL &&
			m_alphaVShader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_alphaUShader.get());
		deps.push_back(m_alphaVShader.get());
		deps.push_back(m_diffuseReflectanceShader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_alphaU.get());
		renderer->unregisterShaderForResource(m_alphaV.get());
		renderer->unregisterShaderForResource(m_diffuseReflectance.get());
	}

void generateCode(std::ostringstream &oss,
	const std::string &evalName,
	const std::vector<std::string> &depNames) const {
	oss << "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
		<< "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
		<< "    if (ds <= 0.0)" << endl
		<< "        return 0.0f;" << endl
		<< "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
		<< "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
		<< "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
		<< "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
		<< "}" << endl
		<< endl
		<< "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
		<< "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
		<< "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
		<< "        return 0.0;" << endl
		<< "    float nDotM = cosTheta(m);" << endl
		<< "    return min(1.0, min(" << endl
		<< "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
		<< "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
		<< "}" << endl
		<< endl
		<< "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
		<< "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
		<< "    	return vec3(0.0);" << endl
		<< "   vec3 H = normalize(wi + wo);" << endl
		<< "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
		<< "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
		<< "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
		<< "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
		<< "   float G = " << evalName << "_G(H, wi, wo);" << endl
		<< "   return reflectance * (D * G / (4*cosTheta(wi)));" << endl
		<< "}" << endl
		<< endl
		<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
		<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
		<< "    	return vec3(0.0);" << endl
		<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
		<< "}" << endl;
}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_specularReflectance;
	ref<const Texture> m_diffuseReflectance;
	
	ref<const Texture> m_alphaU;
	ref<const Texture> m_alphaV;
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_diffuseReflectanceShader;
	ref<Shader> m_alphaUShader;
	ref<Shader> m_alphaVShader;

};

Shader *PreStochasticReflectance::createShader(Renderer *renderer) const {
	return new PreStochasticShader(renderer,
		new ConstantSpectrumTexture(m_reflectance), new ConstantSpectrumTexture(m_diffuseReflectance), 
		new ConstantFloatTexture(m_alphaU), new ConstantFloatTexture(m_alphaV));
}

MTS_IMPLEMENT_CLASS(PreStochasticShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(PreStochasticReflectance, false, BSDF)
MTS_EXPORT_PLUGIN(PreStochasticReflectance, "PreStochastic Reflectance BRDF")
MTS_NAMESPACE_END
