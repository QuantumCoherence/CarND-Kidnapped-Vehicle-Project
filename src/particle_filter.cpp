/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#include "Eigen/Dense"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
using Eigen::VectorXd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (num_particles == 0) num_particles = 20;
	weights.resize(num_particles, 1.0);
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for xmy and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	Particle particle;
	for (int i = 0; i < num_particles; ++i) {
		 particle.id = i;
		 particle.x = dist_x(gen);
		 particle.y = dist_y(gen);
		 particle.theta = dist_theta(gen);
		 particle.weight = weights[particle.id];
		 particle.associations.reserve(20);
		 particle.sense_x.reserve(20);
		 particle.sense_y.reserve(20);
		 particle.landmark_count = 0;
		 particles.push_back(particle);

		 //std::cout << "p.x " << particles[i].x << "\tp.y " << particles[i].y << "\tp.theta " << particles[i].theta << "\tpt.w " << particles[i].weight << "\tp.Ass " << particles[i].associations.size() << std::endl;
	}
	resampled_particles.resize(0);

	is_initialized = true;
	this_part = 0;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for velocity and yaw_rate
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);


	for (int i = 0; i < num_particles; i++){
		if (yaw_rate > 0.0001){ // thetad not null
			particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)- sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		} else { // Thetad null
			particles[i].x += velocity*delta_t*(cos(particles[i].theta));
			particles[i].y += velocity*delta_t*(sin(particles[i].theta));
			particles[i].theta;
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
		//std::cout << "pt.x " << particles[i].x << "\tpt.y " << particles[i].y << "\tp.theta " << particles[i].theta << "\tpt.w " << particles[i].weight << "\tp.Ass " << particles[i].associations.size() << std::endl;

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	int row;
	int col;
	unsigned int obsvs = observations.size();
	if (particles[this_part].associations.size()< obsvs){
		particles[this_part].associations.resize(obsvs);
		particles[this_part].sense_x.resize(obsvs);
		particles[this_part].sense_y.resize(obsvs);
	}
	std::vector<LandmarkObs> loc_observations(obsvs);
	loc_observations=observations;
	int landmks = predicted.size();
	MatrixXd distances = MatrixXd(obsvs, landmks);
	//cout << "P.id " << particles[this_part].id << " part_index " << this_part << endl;;
	for (unsigned int observation = 0 ; observation < obsvs; observation++){
		//cout << "observation " << observations[observation].id << "\tx " << observations[observation].x << "\ty " << observations[observation].y << endl;
		for (int landmark = 0; landmark < landmks; landmark++){
			distances(observation,landmark) =	 dist(observations[observation].x,observations[observation].y, predicted[landmark].x, predicted[landmark].y);
		}
	}
	//for (int l = 0 ; l < landmks;l++) {
	//		cout << "landmark " << predicted[l].id << "\tx " << predicted[l].x << "\ty " << predicted[l].y << endl;
	//}
	//cout << distances << endl << "Distance Matrix " << endl;
	for (unsigned int i=0;i<obsvs;i++) {
		//	cout << "Rows x Cols: " << distances.rows() << " x " << distances.cols() << "\tShortest distance " << distances.minCoeff(&row, &col);
		//	cout << " at row " << row << " col " <<  col << endl;
		//	cout << "Observation Id " << observations[row].id << "\tX " << observations[row].x << "\tY " << observations[row].y << endl;
		//	cout << "Landmark    Id " << predicted[col].id << "\tX " << predicted[col].x << "\tY " << predicted[col].y << endl;
		distances.minCoeff(&row, &col);
		particles[this_part].associations[i] = predicted[col].id;
		particles[this_part].sense_x[i] = loc_observations[row].x;
		particles[this_part].sense_y[i] = loc_observations[row].y;
		particles[this_part].landmark_count = i+1;
		removeRow(distances, row);
		removeColumn(distances, col);
		loc_observations.erase (loc_observations.begin()+row);
		predicted.erase (predicted.begin()+col);
	//	cout << distances << endl;
	}
	//for (unsigned int i=0;i<obsvs;i++){
	//	cout << "ass len " << particles[this_part].associations.size() << "\tID " <<  particles[this_part].associations[i] << "\tx " << particles[this_part].sense_x[i] << "\ty " << particles[this_part].sense_y[i] << endl;
	//}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	unsigned int i;
	vector<LandmarkObs> observed(observations.size());
	vector<LandmarkObs> predicted;
	LandmarkObs landmark;
	cout << endl << "Updateweights " << endl;
	cout << "weights input " << endl;
	for (this_part=0; this_part < particles.size();this_part++){ // particles loop
		//cout << "p " << this_part <<  endl;
		for (i=0;i<observations.size();i++){ // observations loop for current this_part particle
		// convert the current array of observations from car to into map coordinates
			observed[i].x = cos(particles[this_part].theta)*observations[i].x - sin(particles[this_part].theta)*observations[i].y + particles[this_part].x;
			observed[i].y = sin(particles[this_part].theta)*observations[i].x + cos(particles[this_part].theta)*observations[i].y + particles[this_part].y;
			observed[i].id = observations[i].id;
			//cout << "o_x " << observed[i].x << " o_y " << observed[i].y << endl;
		}
		for (i=0;i<map_landmarks.landmark_list.size();i++){ // landmark loop
			// find all landamrks that are within sensor_range for this_part particle
			if (dist(particles[this_part].x, particles[this_part].y, (double)map_landmarks.landmark_list[i].x_f, (double)map_landmarks.landmark_list[i].y_f)<=sensor_range){
				landmark.id = map_landmarks.landmark_list[i].id_i;
				landmark.x = (double)map_landmarks.landmark_list[i].x_f;
				landmark.y = (double)map_landmarks.landmark_list[i].y_f;
				predicted.push_back(landmark);
			}
		}
		//cout << "lndmrks len "  << predicted.size() << endl;
		dataAssociation(predicted, observed);
		predicted.clear();
		cout << "\tw " <<  this_part << "\t" << particles[this_part].weight << endl;
	}
	double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
	cout << "weight out " << endl;
	// calculate exponent

	for (this_part =0; this_part < particles.size();this_part++){
		particles[this_part].weight = 1.0;
		for(i=0; i<particles[this_part].landmark_count; i++){
			double mu_x = (double)map_landmarks.landmark_list[particles[this_part].associations[i]-1].x_f;
			double mu_y = (double)map_landmarks.landmark_list[particles[this_part].associations[i]-1].y_f;
			double x_obs = particles[this_part].sense_x[i];
			double y_obs = particles[this_part].sense_y[i];
			double exponent = ( (x_obs - mu_x)*(x_obs - mu_x) )/( 2*std_landmark[0]*std_landmark[0] ) + ( (y_obs - mu_y)*(y_obs - mu_y) )/( 2*std_landmark[1]*std_landmark[1] );
			particles[this_part].weight *= gauss_norm * exp(-exponent);
			//cout << "ID " << particles[this_part].associations[i] << "\tmu_x " << mu_x << "\tmu_y " << mu_y << "\tx " << x_obs << "\ty " << y_obs << "\texp " << exponent << "\texp(-exponent) " << exp(-exponent) << endl;;
		}
		weights[this_part] = particles[this_part].weight;
		cout << "\tw " <<  this_part << "\t" << particles[this_part].weight << endl;
	}
	cout << "weights vector " << endl;
	for (this_part =0; this_part < particles.size();this_part++){
		particles[this_part].landmark_count = 0;
		cout << weights[this_part] << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// Set of current particles

	random_device rd;     // only used once to initialise (seed) engine
	mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	uniform_int_distribution<int> uni(0,num_particles);
	normal_distribution<double> rand(0, 1.f);

	int index = uni(rng);

	double beta = 0.0;
	double mw =  *max_element(weights.begin(), weights.end());

	for (int i=0; i<num_particles ;i++){
		beta += rand(rng) * 2.0 * mw;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}

		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;
	resampled_particles.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


void ParticleFilter::removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void ParticleFilter::removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}
