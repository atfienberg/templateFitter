//testing the template fitter class

#include "TemplateFitter.hh"

#include "TApplication.h"
#include "TSystem.h"
#include "TFile.h"
#include "TSpline.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TPaveText.h"
#include "TString.h"
#include "TAxis.h"
#include "TF1.h"
#include "TColor.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <string>
#include <numeric>
#include <memory>

using namespace std;

void displayResults(TemplateFitter& tf, TemplateFitter::Output out, 
		    vector<UShort_t> sampleTimes, vector<UShort_t> trace, 
		    string title, TSpline3* tSpline);

int main(){  
  new TApplication("app", 0, nullptr);
  
  //construct 
  unique_ptr<TFile> splineF(new TFile("../adjustedBoardTemplate10_26_15.root"));
  TSpline3* tSpline = (TSpline3*) splineF->Get("masterSpline");
  TemplateFitter tf(tSpline, -10, 90);
  
  unique_ptr<TFile> outf(new TFile("testingOut.root","recreate"));

  gRandom->SetSeed(17);

  //test single pulse fit with flat errors  
  cout << "testing single pulse with flat errors: " << endl;
  double time = 8 + 1*(gRandom->Rndm() - 0.5);
  cout << "true time: " << time << endl;
  double energy = 3000.0 + gRandom->Gaus(0, 0.01*3000);
  cout << "true energy : " << energy << endl;
  double pedestal = 1000.0 + 100*(gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  double noise = 1.65;
  vector<UShort_t> samples(30);
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = pedestal;
    }
    else{
      samples[s] = pedestal + energy * tSpline->Eval(s - time); 
    }
    samples[s] +=  gRandom->Gaus(0, noise);
  }
  double timeGuess = max_element(samples.begin(), samples.end()) - samples.begin(); 
  auto out = tf.fit(samples, timeGuess, noise);  
  vector<UShort_t> sampleTimes(30);
  iota(sampleTimes.begin(),sampleTimes.end(),0.0);
  displayResults(tf, out, sampleTimes, samples, "singlePulseFlatNoise", tSpline);
    
  //single pulse arbitrary noise
  cout << "testing arbitrary sample noise, adding sqrt((s-ped)/50) worth of noise" << endl;
  vector<double> errors(samples.size());
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = pedestal ;
    }
    else{
      samples[s] = pedestal + energy * tSpline->Eval(s - time); 
    }
    double sampleVariance = noise*noise;
    if(samples[s] - pedestal> 0){
      sampleVariance += (samples[s]-pedestal)*(samples[s] - pedestal)/(50*50);
    }
    double sampleUncertainty = sqrt(sampleVariance);
    samples[s] +=  gRandom->Gaus(0, sampleUncertainty);
    errors[s] = sampleUncertainty;
  }
  timeGuess = max_element(samples.begin(), samples.end()) - samples.begin();
  out = tf.fit(samples, timeGuess, errors);
  displayResults(tf, out, sampleTimes, samples, "singlePulseArbitraryNoise", tSpline);

  cout << "testing discontiguous fit regions (clipping) with flat noise: " << endl;
  energy *= 4;
  cout << "new energy: " << energy  << endl;
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = pedestal;
    }
    else{
      samples[s] = pedestal + energy * tSpline->Eval(s - time); 
    }
    samples[s] +=  gRandom->Gaus(0, noise);  
    samples[s] = samples[s] > 4096 ? 4096 : samples[s];
  }

  //find good samples (leave out anything over 4000)
  vector<UShort_t> goodSamples;
  vector<UShort_t> times;
  for(size_t s = 0; s < samples.size(); ++s){
    if(samples[s] < 4000){
      goodSamples.push_back(samples[s]);
      times.push_back(s);
    }
    else{
      cout << "sample " << s << " clipping" << endl;
    }
  }
  timeGuess = max_element(goodSamples.begin(), goodSamples.end()) - goodSamples.begin();
  out = tf.discontiguousFit(goodSamples, times, timeGuess, noise);
  displayResults(tf, out, times, goodSamples, "clippedFitWithFlatErrors", tSpline);
  
  cout << "testing clipped pulse with arbitrary errors" << endl;
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = pedestal ;
    }
    else{
      samples[s] = pedestal + energy * tSpline->Eval(s - time); 
    }
    double sampleVariance = noise*noise;
    if(samples[s] - pedestal> 0){
      sampleVariance += (samples[s]-pedestal)*(samples[s] - pedestal)/(50*50);
    }
    double sampleUncertainty = sqrt(sampleVariance);
    samples[s] +=  gRandom->Gaus(0, sampleUncertainty);
    errors[s] = sampleUncertainty;
    samples[s] = samples[s] > 4096 ? 4096 : samples[s];
  }
  vector<double> goodErrors;
  goodSamples.resize(0);
  times.resize(0);
  for(size_t s = 0; s < samples.size(); ++s){
    if(samples[s] < 4000){
      goodSamples.push_back(samples[s]);
      times.push_back(s);
      goodErrors.push_back(errors[s]);
    }
    else{
      cout << "sample " << s << " clipping, ignoring for fit" << endl;
    }
  }
  timeGuess = max_element(goodSamples.begin(), goodSamples.end()) - goodSamples.begin();
  out = tf.discontiguousFit(goodSamples, times, timeGuess, goodErrors);
  displayResults(tf, out, times, goodSamples, "clippedWithArbitraryErrors", tSpline);

  cout << "testing double pulse fit with flat errors: " << endl;
  energy = energy / 4;
  cout << "True E1 : " << energy << endl;
  double energy2 =  energy * (0.5 + 0.4*(gRandom->Rndm()-0.5));
  cout << "True E2 : " << energy2 << endl;
  cout << "True time 1: " << time << endl;
  double time2 = time + 2.5 + 0.5*(gRandom->Rndm() - 0.5);
  cout << "true time 2: " << time2 << endl;
  cout << "true pedestal : " << pedestal << endl;
  for(std::size_t s = 0; s < samples.size(); ++s) { 
    if((s - time2 < -10) || (s - time > 90) ){
      samples[s] = pedestal;
    }
    else{
      samples[s] = energy * tSpline->Eval(s - time) +
	energy2*tSpline->Eval(s - time2) + pedestal; 
    }
    samples[s] +=  gRandom->Gaus(0, noise);
  }    
  double max =  max_element(samples.begin(), samples.end()) - samples.begin();
  vector<double> tGuesses = {max - 2, max + 2};
  out = tf.fit(samples, tGuesses, noise);
  displayResults(tf, out, sampleTimes, samples, "doublePulseFlatErrors", tSpline);

  cout << "testing three pulse fit, adding a third pulse. " << endl;
  double energy3 = 0.25*(energy + 0.4*(gRandom->Rndm() - 0.5));
  cout << "energy 3: " << energy3 << endl;
  double time3 = time2 + 10 + 4*(gRandom->Rndm() - 0.5);
  cout << "time 3: " << time3 << endl;
  for(std::size_t s = 0; s < samples.size(); ++s){
    if(! ((s - time3 < -10) || (s - time3 > 90) ) ){
      samples[s] += energy3*tSpline->Eval(s - time3);
    }
  }
  tGuesses = { 8, 10.5, 20.5};
  out = tf.fit(samples, tGuesses, noise);
  displayResults(tf, out, sampleTimes, samples, "triplePulseFlatErrors", tSpline);

  outf->Write();
}

void displayResults(TemplateFitter& tf, TemplateFitter::Output out, 
		    std::vector<UShort_t> sampleTimes, std::vector<UShort_t> trace, 
		    string title, TSpline3* tSpline){

  const int nPulses = out.times.size();
  //print to terminal
  cout << endl;
  for(int i = 0; i < nPulses; ++i){
    cout << "t" << i + 1 << ": " << out.times[0] << " +/- " <<
      sqrt(tf.getCovariance(i, i)) << endl;
    cout << "scale" << i + 1 << ": " << out.scales[0] << " +/- " <<
      sqrt(tf.getCovariance(i + nPulses, i + nPulses)) << endl;
  }
  cout << "pedestal: " << out.pedestal << " +/- " << 
    sqrt(tf.getCovariance(2*nPulses, 2*nPulses)) << endl;
  cout << "chi2: " << out.chi2 << std::endl;
  cout << endl;
  cout << "covariance matrix" << endl;
  for(int i = 0; i < 2*nPulses + 1; ++i){
    for(int j = 0; j < 2*nPulses + 1; ++j){
      cout << setw(12) << tf.getCovariance(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;

  //make plot
  unique_ptr<TCanvas> c(new TCanvas((title + "_canvas").c_str(), (title + "_canvas").c_str()));
  
  unique_ptr<TGraph> g(new TGraph(0));
  g->SetTitle(title.c_str());
  for(size_t i = 0; i < trace.size(); ++i){
    g->SetPoint(g->GetN(), sampleTimes[i], trace[i]);
  }

  //room for up to three pulses
  auto templateFunction = [&] (double* x, double* p){
    return p[6] + p[1]*tSpline->Eval(x[0] - p[0]) + p[3]*tSpline->Eval(x[0] - p[2]) +
    (x[0] - p[4] > -10 ? p[5]*tSpline->Eval(x[0] - p[4]) : 0);
  };
  unique_ptr<TF1> func(new TF1("fitFunc", templateFunction, 0, 30, 7));
  func->SetLineColor(kBlack);
  func->SetParameters(vector<double>(7,0).data());
  
  g->SetMarkerStyle(20);
  g->Draw("ap");
  g->GetXaxis()->SetTitle("sample number");
  g->GetYaxis()->SetTitle("ADC counts");
  g->GetYaxis()->SetTitleOffset(1.5);

  double yMin = g->GetYaxis()->GetXmin();
  double yMax = g->GetYaxis()->GetXmax();
  std::unique_ptr<TPaveText> txtbox(new TPaveText(15, yMin + (yMax - yMin)*0.9, 
						  29, yMin + (yMax - yMin) * 0.5));
  txtbox->SetFillColor(kWhite);
  func->SetParameter(6, out.pedestal);
  for(int i = 0; i < nPulses; ++i){
    func->SetParameter(2*i, out.times[i]);
    txtbox->AddText(Form("t_{%i}: %.3f #pm %.3f", i+1, 
			 out.times[i], sqrt(tf.getCovariance(i, i))));
    func->SetParameter(2*i + 1, out.scales[i]);
    txtbox->AddText(Form("E_{%i}: %.0f #pm %.0f", i + 1,
			 out.scales[i], sqrt(tf.getCovariance(nPulses + i, nPulses + i))));    
  }
  txtbox->AddText(Form("pedestal: %.0f #pm %.1f", 
		       out.pedestal, sqrt(tf.getCovariance(2*nPulses, 2*nPulses))));
  txtbox->AddText(Form("#chi^{2} / NDF : %.2f", out.chi2));
  
  std::vector<unique_ptr<TF1>> components;
  if(nPulses > 1){
    int colors[3] = {kRed, kBlue, kMagenta + 2};
    for(int i = 0; i < nPulses; ++i){
      components.emplace_back(new TF1("fitFunc", templateFunction, 0, 30, 7));
      components.back()->SetParameters(vector<double>(7,0).data());
      components.back()->SetParameter(6, out.pedestal);
      components.back()->SetParameter(2*i, out.times[i]);
      components.back()->SetParameter(2*i + 1, out.scales[i]);
      components.back()->SetLineColor(colors[i]);
      components.back()->SetNpx(1000);
      components.back()->Draw("same");
    }
  }
      
  func->SetNpx(1000);
  func->Draw("same");
  txtbox->Draw("same");
  c->Print((title + ".pdf").c_str());
  c->Write(); 

  c->Modified();
  c->Update();
  c->Draw();
  gSystem->ProcessEvents();
  

  cout << title << " displayed. Any key to move on..." << endl;
  cin.ignore();
}  
