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

  gRandom->SetSeed(0);

  //test separate even odd fit
  cout << "testing even odd fit: " << endl;
  double time = 8 + 1*(gRandom->Rndm() - 0.5);
  cout << "true time: " << time << endl;
  double energy = 3000.0 + gRandom->Gaus(0, 0.01*3000);
  cout << "true energy : " << energy << endl;
  double pedestal = 1000.0 + 100*(gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  double noise = 1.65;
  double evenEnergy = energy*1.05;
  double oddEnergy = energy*0.95;
  double evenPedestal = pedestal + 100;
  double oddPedestal = pedestal - 100;
  cout << "even scale : " << evenEnergy << endl;
  cout << "odd scale : " << oddEnergy << endl;
  cout << "even pedestal : " << evenPedestal << endl;
  cout << "odd pedestal : " << oddPedestal << endl;
  vector<UShort_t> samples(32);
  vector<UShort_t> sampleTimes(32);
  iota(sampleTimes.begin(), sampleTimes.end(), 0);
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = s % 2 == 0 ? evenPedestal : oddPedestal;
    }
    else{
      samples[s] = s % 2 == 0 ? 
	evenPedestal + evenEnergy * tSpline->Eval(s - time) :
	oddPedestal + oddEnergy * tSpline->Eval(s - time);
    }
    samples[s] +=  gRandom->Gaus(0, noise);
  }
  double timeGuess = std::max_element(samples.begin(), samples.end()) - samples.begin();
  auto out = tf.fit(samples, timeGuess, noise);

  displayResults(tf, out, sampleTimes, samples, "evenOddFit", tSpline);
  
  //different noise on even and odd samples 
  auto flatErrorSamples = samples;
  vector<double> errors(samples.size());
  for(int i = 0; i < 32; ++i){
    errors[i] = i % 2 == 0 ? 3 : 1;
  }
  for(size_t s = 0; s < samples.size(); ++s) { 
    if((s - time < -10) || (s - time > 90) ){
      samples[s] = s % 2 == 0 ? evenPedestal : oddPedestal;
    }
    else{
      samples[s] = s % 2 == 0 ? 
	evenPedestal + evenEnergy * tSpline->Eval(s - time) :
	oddPedestal + oddEnergy * tSpline->Eval(s - time);
    }
    samples[s] +=  gRandom->Gaus(0, errors[s]);
  }
  cout << "try arbitrary errors, in this case different errors on even and odd samples" << endl;
  out = tf.fit(samples, timeGuess, errors);
  displayResults(tf, out, sampleTimes, samples, "evenOddFitArbErrors", tSpline);
  
  //try a double fit
  cout << "try double fit with flat errors: " << endl;
  double time2 = time + 6 + 2*(1-gRandom->Rndm());
  cout << "time2: " << time2 << endl;
  double energy2 = energy/2.0 + 500*(1-gRandom->Rndm());
  double evenEnergy2 = energy2*1.05;
  double oddEnergy2 = energy2*0.95;
  cout << "even scale 2: " << evenEnergy2 << endl;
  cout << "odd scale 2: " << oddEnergy2 << endl;
  samples = flatErrorSamples;
  for(size_t s = 0; s < samples.size(); ++s) { 
    if(! ((s - time2 < -10) || (s - time2 > 90) )){
      samples[s] += s % 2 == 0 ? 
	evenEnergy2 * tSpline->Eval(s - time2) :
	oddEnergy2 * tSpline->Eval(s - time2);
    }
  }
  out = tf.fit(samples, vector<double>{8, 14}, noise);
  displayResults(tf, out, sampleTimes, samples, "evenOddFitDouble", tSpline);
  

}

void displayResults(TemplateFitter& tf, TemplateFitter::Output out, 
		    vector<UShort_t> sampleTimes, vector<UShort_t> trace, 
		    string title, TSpline3* tSpline){

  const int nPulses = out.times.size();
  const TemplateFitter::LinearParams evenOddParams[2] = {out.evenParams, out.oddParams};
  string evenOddString[2] = {"even", "odd"};
  
  //print to terminal
  cout << endl;
  for(int i = 0; i < nPulses; ++i){
    cout << "t" << i + 1 << ": " << out.times[0] << " +/- " <<
      sqrt(tf.getCovariance(i, i)) << endl;
  }
  
  for(int i = 0; i < 2; ++i){
    for(int j = 0; j < nPulses; ++j){
      cout << evenOddString[i] << " scale " << j << " : " <<
	evenOddParams[i].scales[j] << " +/- " << 
	sqrt(tf.getCovariance(nPulses + i*(nPulses+1) + j, nPulses + i*(nPulses+1) + j)) <<
	endl;
    }
    int pedIndex = i == 0 ? 2*nPulses : 3*nPulses+1;
    cout << evenOddString[i] << " pedestal: " << evenOddParams[i].pedestal << " +/- " << 
      sqrt(tf.getCovariance(pedIndex, pedIndex)) << endl;
  }
  cout << "chi2: " << out.chi2 << endl;
  cout << endl;
  cout << "covariance matrix" << endl;
  for(int i = 0; i < 3*nPulses + 2; ++i){
    for(int j = 0; j < 3*nPulses + 2; ++j){
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
    if( static_cast<int>(floor(x[0] + 0.5)) % 2 == 0) {
      return p[6] + p[1]*tSpline->Eval(x[0] - p[0]) + p[3]*tSpline->Eval(x[0] - p[2]) +
      (x[0] - p[4] > -10 ? p[5]*tSpline->Eval(x[0] - p[4]) : 0);
    }
    else{
      return p[7] + p[8]*tSpline->Eval(x[0] - p[0]) + p[9]*tSpline->Eval(x[0] - p[2]) +
      (x[0] - p[4] > -10 ? p[10]*tSpline->Eval(x[0] - p[4]) : 0);
    }
  };
  unique_ptr<TF1> func(new TF1("fitFunc", templateFunction, 0, 32, 11));
  func->SetLineColor(kBlack);
  func->SetParameters(vector<double>(11,0).data());
  
  g->SetMarkerStyle(20);
  g->Draw("ap");
  g->GetXaxis()->SetTitle("sample number");
  g->GetYaxis()->SetTitle("ADC counts");
  g->GetYaxis()->SetTitleOffset(1.5);

  double yMin = g->GetYaxis()->GetXmin();
  double yMax = g->GetYaxis()->GetXmax();
  unique_ptr<TPaveText> txtbox(new TPaveText(18, yMin + (yMax - yMin)*0.9, 
						  31, yMin + (yMax - yMin) * 0.5));
  txtbox->SetFillColor(kWhite);
  func->SetParameter(6, evenOddParams[0].pedestal);
  func->SetParameter(7, evenOddParams[1].pedestal);
  for(int i = 0; i < nPulses; ++i){
    func->SetParameter(2*i, out.times[i]);
    txtbox->AddText(Form("t_{%i}: %.3f #pm %.3f", i+1, 
			 out.times[i], sqrt(tf.getCovariance(i, i))));
  }
  for(int i = 0; i < 2; ++i){
    for(int j = 0; j < nPulses; ++j){
      func->SetParameter(i == 0 ? 2*j + 1 : 8 + j, 
			 evenOddParams[i].scales[j]);
      txtbox->AddText(Form("%s E_{%i}: %.0f #pm %.0f", 
			   evenOddString[i].c_str(),
			   j + 1,
			   evenOddParams[i].scales[j], 
			   sqrt(tf.getCovariance(nPulses + i*(nPulses+1) + j, 
						 nPulses + i*(nPulses+1) + j))));
    }
    int pedIndex = i == 0 ? 2*nPulses : 3*nPulses+1;
    txtbox->AddText(Form("%s pedestal: %.0f #pm %.1f", 
			 evenOddString[i].c_str(),
			 evenOddParams[i].pedestal, 
			 sqrt(tf.getCovariance(pedIndex, pedIndex))));
  }

  txtbox->AddText(Form("#chi^{2} / NDF : %.2f", out.chi2));
        
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
