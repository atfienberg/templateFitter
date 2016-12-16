#include "TemplateFitter.hh"

#include "TFile.h"
#include "TSpline.h"
#include "TRandom.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <iostream>

typedef struct{
  double time;
  double energy;
  double chi2;
} result;

using namespace std;

namespace {
  vector<vector<TemplateFitter::Output>> outs;
}

void fit(TemplateFitter tf, vector<UShort_t> samples, 
  double timeGuess, double noise, vector<TemplateFitter::Output>& out_vect){
  out_vect.resize(100000);
  for(int i = 0; i < 100000; ++i){
    out_vect[i] = tf.fit(samples, timeGuess, noise);
  }
}

int main(){

  TFile* splineF = new TFile("../adjustedBoardTemplate10_26_15.root");
  TSpline3* tSpline = (TSpline3*) splineF->Get("masterSpline");

  TemplateFitter tf(tSpline, -10, 90);
  tf.setAccuracy(1e-2);
  
  gRandom->SetSeed(19);
  //  tf.setAccuracy(1e-6);
  double time = 50 + 10*(gRandom->Rndm() - 0.5);
  cout << "time: " << time << endl;
  double energy = 3000.0 + gRandom->Gaus(0, 0.01*3000);
  cout << "energy : " << energy << endl;
  double pedestal = 1000.0 + 100*(gRandom->Rndm() - 0.5);
  cout << "pedestal " << pedestal << endl;

  double noise = 1.65;
  vector<UShort_t> samples(100);
  for(size_t s = 0; s < 100; ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = 0;
    }
    else{
      samples[s] = energy * tSpline->Eval(s - time); 
    }
    samples[s] +=  gRandom->Gaus(0, noise) + pedestal;
  }
  auto maxiter = max_element(samples.begin(), samples.end());
  int fitStart = maxiter - 5 - samples.begin();
  rotate(samples.begin(), maxiter-5, samples.end());
  samples.resize(42);
  
  double timeGuess = 5;
  
  // TemplateFitter::Output out;
  auto t1 = chrono::high_resolution_clock::now();

  vector<thread> threads;

  outs.resize(8);
  for(int i = 0; i < 8; ++i){
    threads.push_back(thread(fit, tf, samples, timeGuess, noise, ref(outs[i])));
  }

  for(auto& t: threads){
    t.join();
  }
  auto t2 = chrono::high_resolution_clock::now();
  cout << 800000/chrono::duration<double>(t2-t1).count() << " fits per sec " << endl;
  cout << 800000/chrono::duration<double>(t2-t1).count()/4.0 << " fits per sec over 4" << endl;

  auto out = outs.back().back();

  cout << "t: " << out.times[0] + fitStart << endl;
  cout << "scale: " << out.scales[0] << endl;
  cout << "chi2: " << out.chi2 << endl;

  // cout << " t = " << out.times[0] + fitStart << " +/- " << sqrt(tf.getCovariance(0,0)) << endl;
  // cout << " s = " << out.scales[0] << " +/- " << sqrt(tf.getCovariance(1,1)) << endl;
  // cout << " p = " << out.pedestal << " +/- " << sqrt(tf.getCovariance(2,2)) << endl;
  // cout << " chi2: " << out.chi2 << endl;
  
  // cout << "cov" <<endl;
  // for(int i = 0; i < 3; ++i){
  //   for(int j = 0; j < 3; ++j){
  //     cout << tf.getCovariance(i, j) << " ";
  //   }
  //   cout << endl;
  // }
    

}
