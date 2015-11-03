#include "TemplateFitter.hh"

#include "TFile.h"
#include "TSpline.h"
#include "TRandom.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace std;

int main(){

  TFile* splineF = new TFile("../adjustedBoardTemplate10_26_15.root");
  TSpline3* tSpline = (TSpline3*) splineF->Get("masterSpline");

  TemplateFitter tf(tSpline, -10, 90);
  
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
  for(std::size_t s = 0; s < 100; ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = 0;
    }
    else{
      samples[s] = energy * tSpline->Eval(s - time); 
    }
    samples[s] +=  gRandom->Gaus(0, noise) + pedestal;
  }
  auto maxiter = std::max_element(samples.begin(), samples.end());
  int fitStart = maxiter - 5 - samples.begin();
  std::rotate(samples.begin(), maxiter-5, samples.end());
  samples.resize(30);
  
  double timeGuess = 5;
  
  std::vector<double> errorvect(30, noise);
  TemplateFitter::Output out;
  auto t1 = clock();
  for(int i = 0; i < 1000000; ++i){
    out = tf.fit(samples, timeGuess , noise);
  }
  std::cout << 1000000/(float(clock() - t1)/CLOCKS_PER_SEC) << " fits per sec " << std::endl;
  std::cout << " t = " << out.times[0] + fitStart << " +/- " << std::sqrt(tf.getCovariance(0,0)) << std::endl;
  std::cout << " s = " << out.scales[0] << " +/- " << std::sqrt(tf.getCovariance(1,1)) << std::endl;
  std::cout << " p = " << out.pedestal << " +/- " << std::sqrt(tf.getCovariance(2,2)) << std::endl;
  std::cout << " chi2: " << out.chi2 << std::endl;
  
  std::cout << "cov" <<std::endl;
  for(int i = 0; i < 3; ++i){
    for(int j = 0; j < 3; ++j){
      std::cout << tf.getCovariance(i, j) << " ";
    }
    std::cout << std::endl;
  }
    

}
