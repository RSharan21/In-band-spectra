# MSP in-band-spectra for uGMRT
It calculates the in-band spectra (flux density estimated for chunks of frequency over the wide observing band, where the user can decide on the width of the frequency chunk in terms of the number of channels) on uGMRT data and fits the optimal number of broken power law components using AICc model selection criteria.

## Installation


## Usage
User can run the function in `apply_code_plot.py` in the following manner:
```
import apply_code_plot

apply_code_plot.plot_in_band_spectra(file_name, Sky_temp, number_of_channels_per_chunk, sigma, thres, observing_beam, Primary_component, Show_plots, allow_fit, save_plot, number_of_antennas, receive_output)
```

Following are the brief description of the arguments used in the above code:

```file_name``` : input : str, is the name of the pfd/ar folded data cube.

```Sky_temp``` : input : float, is the sky temperature of the sky at the central frequency of the wideband telescope and the RA, DEC of the pulsar.

```number_of_channels_per_chunk```: input : int, number of channel to be taken in each in-band spectra frequency chunk.

```sigma```: input : float, is minimum value of the parameter p as mentioned in the paper (refer to the paper) above which the phase bins are considered to be ON bins.

```thres```: input : float, is the minimum SNR of the profile per chunk (i.e. any frequency chunk where the profile SNR goes below the ```thres```, that chunk is not plot in the final plot of in-band spectra).

```observing_beam``` : input : str, is the beam former configuration of the observation (i.e. IA or PA beam for uGMRT).

```Primary_component```: input : bool, whether the user want the in-band spectra of the main profile component (this algorithm is not that robust but works where the main and the other components of profile are disconnected).

```Show_plots```: input : bool, whether the user wants to see the plot or not.

```allow_fit``` : input : bool, whether the user wants to fits the optimal number of broken power laws.

```save_plot``` : input : bool, whether the user wants to save the plots.

```number_of_antennas``` : input : int, number of antennas used in the observation.

```receive_output```  : input : bool, if user wants to check the additional information.

### Example usage with the sample pfd (attached in the example_data folder)
```
import apply_code_plot
cd Example/
f = 'J1120-3618_pa_500_200_4096_4_1_8_16nov2018.raw_PSR_1120-3618.pfd'
apply_code_plot.plot_in_band_spectra(f, 23.0, 15, 5, 3, 'PA', True, True, True, False, 24, False)
```

## Credits
The algorithm are described in the paper Sharan et al. 2024 (submitted to ApJ). <!--If someone uses the code, kindly refer to the paper Sharan et al. 2024 (submitted to ApJ).-->

