Look Ahead DFP and PCS Tutorial — GitHub Project



DFP: Decouple From Present; PCS: Peak Correlation Shift.



\####

Overview:



DFP and PCS provide generic frameworks for solving prediction problems while simultaneously accommodating specific, practically relevant research priorities and objectives.



The Look Ahead Tutorial is an R-based project comprising a collection of exercises and case studies designed to introduce users to — and provide hands-on experience with — the DFP and PCS frameworks.



\####

Author: Marc Wildi — https://marcwildi.com



Repository: https://github.com/wiaidp/R-Package-Look-Ahead



Background (references \& links): https://github.com/wiaidp/R-Package-Look-Ahead/blob/main/about.md



\####

Project Structure:



The project directory is organized into sub-folders:



1.Data

2.Look-Ahead Tutorials

3.Papers

4.R (repository of DFP and PCS optimization)

5.R utility functions



\####



Getting started: Download the Look-Ahead GitHub project to a local folder. In this folder, open the R project by clicking the project icon Look-Ahead\_project.Rproj. This will launch the project in RStudio and automatically set paths to functions and data. From there, load any tutorial file from the `Look-Ahead Tutorials' sub-folder and run the code. Tutorials are separated into DFP and PCS and arranged in order of increasing complexity.





\################################################################################################################

Background:

\################################################################################################################



DFP and PCS are built to address the prediction ATS Trilemma. Forecasting inherently involves three partly competing goals:



I. Accuracy — correctly predicting future levels or signs



II. Timeliness — avoiding undue delays or premature signals



III. Smoothness — suppressing spurious noise and erratic fluctuations



Together, these constitute the ATS Trilemma.





\####

Optimization Principles:

DFP and PCS address the dimensions of the trilemma as follows:



Accuracy: optimized via target correlation and MSE (mean-squared error).



Smoothness: this component is not addressed explicitly (M-SSA and MDFA address smoothness).



Timeliness: addressed explicitly in various forms (decoupling, time-shift at frequency zero, location of peak of Cross Correlation Function (CCF)).



DFP and PCS are specialized to the AT-Dilemma (extensions to include the Smoothness part are under investigation: link with M-SSA).



\####

Looking Ahead



In many applications, the classic MSE paradigm is stuck at the present: most weight is assigned to the last observation, the predictor merely tracks the current data point, and increasing the forecast horizon does not improve the lead. That is, the MSE predictor cannot effectively look ahead and anticipate future dynamics.



DFP addresses this limitation by decoupling the predictor from the last observation, allowing it to look ahead — albeit at the cost of increased MSE at the forecast horizon. The DFP minimises this loss subject to a prescribed decoupling constraint. Notably, decoupling can be linked to a natural measure of lead: the time-shift at frequency zero. In this sense, DFP maximises tracking accuracy at horizon h subject to a given time-shift at frequency zero: a controlled lead or left-shift against the classic MSE predictor.



The time-shift at frequency zero, tau, quantifies the lead of the DFP predictor over the classic MSE predictor as follows: a linear trend is left-shifted by tau time units relative to the MSE benchmark. In practice, this left-shift typically — though not always — extends to adjacent frequencies by continuity. For example, low-frequency business-cycle components may also experience a left-shift, but generally to a lesser degree than tau: the lead achieved at frequency zero carries over to business-cycle frequencies, albeit attenuated.





PCS addresses an aggregate lead measure that accounts for the time-shift across the entire frequency band, rather than only locally at frequency zero. The stuck at the present problem of the classic MSE predictor manifests in the cross-correlation function (CCF): the CCF of the MSE predictor with the target series peaks at lag k = 0, whereas the CCF of an ideal forward-looking predictor would peak at the forecast horizon h. PCS aims precisely at shifting this peak from 0 to h. The CCF thus serves as an aggregate measure of lead (advancement, left-shift) or lag (retardation, right-shift), summarising the cumulative effect of the time-shift of the PCS predictor across the full frequency band.



\####

Efficient Frontier and Pareto Optimality:

In a DFP (or PCS) optimized predictor, any gain in lead inevitably incurs a loss in target correlation or MSE at the forecast horizon h. There is no free lunch.



This trade-off is a direct consequence of DFP (and PCS) residing on the efficient frontier of the Accuracy-Timeliness (AT) Dilemma (Pareto optimality):



Classical Max-Likelihood (MSE) predictors represent a single point on this frontier.



DFP and PCS extend the solution space to the full frontier, offering a richer and more flexible set of forecasting solutions.



\####

What Makes DFP/PCS Distinctive



A. Generality and Customisation

Classical linear forecasting methods emerge as special cases of DFP/PCS, which can then be refined to reflect specific research priorities and objectives. The tutorial demonstrates this customisation for two canonical problems: classic ARMA forecasting and business-cycle filtering. Particular emphasis is placed on difficult forecasting problems in which the MSE predictor is stuck at the present — settings where a right-shift of the CCF is difficult or even impossible to achieve.



B. Interpretability

The optimisation criteria are grounded in clear, fundamental principles, yielding closed-form solutions that are uniquely determined and straightforward to interpret and communicate.



C. Transparency

Unlike black-box methods, DFP/PCS provide direct insight into the forecasting mechanism: optimisation is performed in closed form and leads to unique, fully traceable solutions.



Together, these qualities make DFP/PCS especially well-suited for settings where opacity is either prohibited — such as compliance-driven or regulatory environments — or simply undesirable, such as when a deeper understanding of the underlying forecasting logic is required.



\####

Alternative Prediction Packages:

The author proposes the following complementary R-based prediction frameworks (https://marcwildi.com):



1. MDFA: https://github.com/wiaidp/MDFA-tutorial. MDFA is a generic prediction approach addressing the full ATS trilemma in the frequency domain. Accuracy, smoothness, and timeliness are derived from and defined on the amplitude and phase characteristics of the predictor filter.
2. M-SSA: https://github.com/wiaidp/R-package-SSA-Predictor. The M-SSA specializes in the Accuracy-Smoothness trade-off, targeting applications where the cost of spurious false alarms is prohibitive.





\####

Positioning of the Approaches:



\-MDFA is the most general, all-round framework — operating in the frequency domain and addressing all three ATS dimensions simultaneously.



\-M-SSA and DFP/PCS are formulated in the time domain and specialize in targeted aspects of the prediction problem, making them more focused than MDFA.



\-Their domain-specific optimization principles may offer greater intuitive appeal than frequency-domain statistics for certain users and applications.



\-Current research aims at unifying M-SSA and DFP/PCS to combine their respective strengths into a single, more comprehensive time-domain prediction framework.

