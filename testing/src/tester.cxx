// testing the template fitter class

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
                    string title, TSpline3* tSplineA, TSpline3* tSplineB);

int main() {
  new TApplication("app", 0, nullptr);

  // construct
  TFile splineF("../beamTemplateFile37.root");
  TSpline3* tSpline = (TSpline3*)splineF.Get("masterSpline");
  TFile splineF2("../beamleakTemplateFile37.root");
  TSpline3* tSplineB = (TSpline3*)splineF2.Get("masterSpline");
  TemplateFitter tf(tSpline, tSplineB, -10, 90);

  unique_ptr<TFile> outf(new TFile("testingOut.root", "recreate"));

  gRandom->SetSeed(17);

  // test single pulse fit with flat errors
  cout << "testing single pulse with flat errors A: " << endl;
  double time = 8 + 1 * (gRandom->Rndm() - 0.5);
  cout << "true time: " << time << endl;
  double energy = 3000.0 + gRandom->Gaus(0, 0.01 * 3000);
  cout << "true energy : " << energy << endl;
  double pedestal = 1000.0 + 100 * (gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  double noise = 1.65;
  vector<UShort_t> samples(45);
  for (size_t s = 0; s < samples.size(); ++s) {
    if ((s - time < -10) || (s - time > 90)) {
      samples[s] = pedestal;
    } else {
      samples[s] = pedestal + energy * tSpline->Eval(s - time);
    }
    samples[s] += gRandom->Gaus(0, noise);
  }
  double timeGuess =
      max_element(samples.begin(), samples.end()) - samples.begin();
  auto out = tf.fit(samples, timeGuess, noise);
  std::vector<unsigned short> sampleTimes(samples.size());
  std::iota(sampleTimes.begin(), sampleTimes.end(), 0.0);
  displayResults(tf, out, sampleTimes, samples, "template A only", tSpline,
                 tSplineB);

  cout << "testing single pulse with flat errors B: " << endl;
  time = 8 + 1 * (gRandom->Rndm() - 0.5);
  cout << "true time: " << time << endl;
  energy = 3000.0 + gRandom->Gaus(0, 0.01 * 3000);
  cout << "true energy : " << energy << endl;
  pedestal = 1000.0 + 100 * (gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  noise = 1.65;
  for (size_t s = 0; s < samples.size(); ++s) {
    if ((s - time < -10) || (s - time > 90)) {
      samples[s] = pedestal;
    } else {
      samples[s] = pedestal + energy * tSplineB->Eval(s - time);
    }
    samples[s] += gRandom->Gaus(0, noise);
  }
  timeGuess = max_element(samples.begin(), samples.end()) - samples.begin();
  out = tf.fit(samples, timeGuess, noise);
  displayResults(tf, out, sampleTimes, samples, "template B only", tSpline,
                 tSplineB);

  cout << "testing single pulse with flat errors half and half: " << endl;
  time = 8 + 1 * (gRandom->Rndm() - 0.5);
  cout << "true time: " << time << endl;
  energy = 3000.0 + gRandom->Gaus(0, 0.01 * 3000);
  cout << "true energy : " << energy << endl;
  pedestal = 1000.0 + 100 * (gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  noise = 1.65;
  for (size_t s = 0; s < samples.size(); ++s) {
    if ((s - time < -10) || (s - time > 90)) {
      samples[s] = pedestal;
    } else {
      samples[s] = pedestal + energy / 2.0 * tSpline->Eval(s - time) +
                   energy / 2.0 * tSplineB->Eval(s - time);
    }
    samples[s] += gRandom->Gaus(0, noise);
  }
  timeGuess = max_element(samples.begin(), samples.end()) - samples.begin();
  out = tf.fit(samples, timeGuess, noise);
  displayResults(tf, out, sampleTimes, samples, "half and half", tSpline,
                 tSplineB);

  cout << "testing double pulse with flat errors, A then B: " << endl;
  time = 8 + 1 * (gRandom->Rndm() - 0.5);
  cout << "true time 1: " << time << endl;
  energy = 3000.0 + gRandom->Gaus(0, 0.01 * 3000);
  cout << "true energy 1: " << energy << endl;
  double time2 = 16 + 1 * (gRandom->Rndm() - 0.5);
  cout << "true time 2: " << time2 << endl;
  double energy2 = 4500.0 + gRandom->Gaus(0, 0.01 * 3000);
  cout << "true energy 2: " << energy2 << endl;
  pedestal = 1000.0 + 100 * (gRandom->Rndm() - 0.5);
  cout << "true pedestal " << pedestal << endl;
  noise = 1.65;
  for (size_t s = 0; s < samples.size(); ++s) {
    if ((s - time < -10) || (s - time > 90)) {
      samples[s] = pedestal;
    } else {
      samples[s] = pedestal + energy * tSpline->Eval(s - time) +
                   energy2 * tSplineB->Eval(s - time2);
    }
    samples[s] += gRandom->Gaus(0, noise);
  }

  out = tf.fit(samples, std::vector<double>{8, 16}, noise);
  displayResults(tf, out, sampleTimes, samples, "first A, second B", tSpline,
                 tSplineB);

  outf->Write();
}

void displayResults(TemplateFitter& tf, TemplateFitter::Output out,
                    std::vector<UShort_t> sampleTimes,
                    std::vector<UShort_t> trace, string title,
                    TSpline3* tSplineA, TSpline3* tSplineB) {
  const int nPulses = out.times.size();
  int min = *std::min_element(sampleTimes.begin(), sampleTimes.end());
  int max = *std::max_element(sampleTimes.begin(), sampleTimes.end());
  // print to terminal
  cout << endl;
  std::vector<double> sigA;
  std::vector<double> sigB;
  std::vector<double> sigT;
  std::vector<double> sigTot;

  for (int i = 0; i < nPulses; ++i) {
    sigT.push_back(sqrt(tf.getCovariance(i, i)));
    cout << "t" << i + 1 << ": " << out.times[i] << " +/- " << sigT.back()
         << endl;

    double varEA = tf.getCovariance(2 * i + nPulses, 2 * i + nPulses);
    sigA.push_back(sqrt(varEA));
    cout << "energyA" << i + 1 << ": " << out.scalesA[i] << " +/- "
         << sigA.back() << endl;

    double varEB = tf.getCovariance(2 * i + 1 + nPulses, 2 * i + 1 + nPulses);
    sigB.push_back(sqrt(varEB));
    cout << "energyB" << i + 1 << ": " << out.scalesB[i] << " +/- "
         << sigB.back() << endl;

    double correlation = tf.getCovariance(2 * i + nPulses, 2 * i + 1 + nPulses);
    sigTot.push_back(sqrt(varEA + varEB + 2 * correlation));
    cout << "total energy: " << out.scalesA[i] + out.scalesB[i] << " +/- "
         << sigTot.back() << std::endl;
  }
  double sigPed = sqrt(tf.getCovariance(3 * nPulses, 3 * nPulses));
  cout << "pedestal: " << out.pedestal << " +/- " << sigPed << endl;
  cout << "chi2: " << out.chi2 << std::endl;
  cout << endl;
  cout << "covariance matrix" << endl;
  for (int i = 0; i < 3 * nPulses + 1; ++i) {
    for (int j = 0; j < 3 * nPulses + 1; ++j) {
      cout << setw(12) << tf.getCovariance(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;

  // make plot
  unique_ptr<TCanvas> c(
      new TCanvas((title + "_canvas").c_str(), (title + "_canvas").c_str()));

  unique_ptr<TGraph> g(new TGraph(0));
  g->SetTitle(title.c_str());
  for (size_t i = 0; i < trace.size(); ++i) {
    g->SetPoint(g->GetN(), sampleTimes[i], trace[i]);
  }

  // template function
  auto templateFunction = [&](double* x, double* p) {
    double pedestal = p[0];
    double pulseVal = pedestal;
    double t = x[0];
    for (int pulse = 0; pulse < nPulses; ++pulse) {
      double* pulseParams = p + 1 + 3 * pulse;
      double tp = t - pulseParams[0];
      if (tp > -10 && tp < 90) {
        pulseVal += pulseParams[1] * tSplineA->Eval(tp) +
                    pulseParams[2] * tSplineB->Eval(tp);
      }
    }
    return pulseVal;
  };
  unique_ptr<TF1> func(
      new TF1("fitFunc", templateFunction, min, max, 3 * nPulses + 1));
  func->SetLineColor(kMagenta + 2);
  func->SetParameter(0, out.pedestal);
  for (int i = 0; i < nPulses; ++i) {
    func->SetParameter(1 + 3 * i, out.times[i]);
    func->SetParameter(1 + 3 * i + 1, out.scalesA[i]);
    func->SetParameter(1 + 3 * i + 2, out.scalesB[i]);
  }

  g->SetMarkerStyle(20);
  g->Draw("ap");
  g->GetXaxis()->SetTitle("sample number");
  g->GetYaxis()->SetTitle("ADC counts");
  g->GetYaxis()->SetTitleOffset(1.5);
  g->GetXaxis()->SetRangeUser(min, max);
  func->SetNpx(10000);
  func->Draw("same");

  double yMin = g->GetYaxis()->GetXmin();
  double yMax = g->GetYaxis()->GetXmax();
  std::unique_ptr<TPaveText> txtbox(new TPaveText(
      22, yMin + (yMax - yMin) * 0.9, 43, yMin + (yMax - yMin) * 0.5));
  txtbox->SetFillColor(kWhite);
  for (int i = 0; i < nPulses; ++i) {
    txtbox->AddText(
        Form("EA_{%i}: %.0f #pm %.0f", i + 1, out.scalesA[i], sigA[i]));
    txtbox->AddText(
        Form("EB_{%i}: %.0f #pm %.0f", i + 1, out.scalesB[i], sigB[i]));
    txtbox->AddText(Form("ETOTAL_{%i}: %.0f #pm %.0f", i + 1,
                         out.scalesB[i] + out.scalesA[i], sigTot[i]));
  }
  txtbox->AddText(Form("pedestal: %.0f #pm %.1f", out.pedestal, sigPed));
  txtbox->AddText(Form("#chi^{2} / NDF : %.2f", out.chi2));

  txtbox->Draw("same");
  c->Print((title + ".pdf").c_str());
  c->Print((title + ".png").c_str());
  c->Write();

  c->Modified();
  c->Update();
  c->Draw();
  gSystem->ProcessEvents();

  cout << title << " displayed. Any key to move on..." << endl;
  cin.ignore();
}
