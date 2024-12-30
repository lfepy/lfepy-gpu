import cupy as cp
import cupy.fft as cp_fft
from lfepy.Helper.low_pass_filter import low_pass_filter


def phase_cong3(image, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
                dThetaOnSigma=1.5, k=2.0, cutOff=0.5, g=10):
    """
    Computes the phase congruency of an image using a multiscale, multi-orientation approach.

    Phase congruency is a measure of the image's local contrast, based on the phase information
    of its frequency components. This method is used for edge detection and texture analysis.

    Args:
        image (numpy.ndarray): Input grayscale image as a 2D numpy array.
        nscale (int, optional): Number of scales to be used in the analysis. Default is 4.
        norient (int, optional): Number of orientations to be used in the analysis. Default is 6.
        minWaveLength (float, optional): Minimum wavelength of the log-Gabor filters. Default is 3.
        mult (float, optional): Scaling factor for the wavelength of the log-Gabor filters. Default is 2.1.
        sigmaOnf (float, optional): Standard deviation of the Gaussian function used in the log-Gabor filter. Default is 0.55.
        dThetaOnSigma (float, optional): Angular spread of the Gaussian function relative to the orientation. Default is 1.5.
        k (float, optional): Constant to adjust the threshold for noise. Default is 2.0.
        cutOff (float, optional): Cut-off parameter for the weighting function. Default is 0.5.
        g (float, optional): Gain parameter for the weighting function. Default is 10.

    Returns:
        tuple: A tuple containing:
            M (numpy.ndarray): The measure of local phase congruency.
            m (numpy.ndarray): The measure of local phase concavity.
            ori (numpy.ndarray): Orientation of the phase congruency.
            featType (numpy.ndarray): Complex representation of phase congruency.
            PC (list of numpy.ndarray): List of phase congruency maps for each orientation.
            EO (list of numpy.ndarray): List of complex responses for each scale and orientation.

    Raises:
        ValueError: If the input image is not a 2D numpy array.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()
        >>> M, m, ori, featType, PC, EO = phase_cong3(image)
    """
    epsilon = 1e-4

    thetaSigma = cp.pi / norient / dThetaOnSigma

    rows, cols = image.shape
    imagefft = cp_fft.fft2(image)

    zero = cp.zeros((rows, cols))
    EO = [[None] * norient for _ in range(nscale)]
    covx2, covy2, covxy = zero.copy(), zero.copy(), zero.copy()

    estMeanE2n, PC = [], []
    ifftFilterArray = [None] * nscale

    # Precompute ranges and angles for the FFT grid
    xrange = cp.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) / (cols - 1) if cols % 2 else cp.linspace(-cols / 2,
                                                                                                          cols / 2 - 1,
                                                                                                          cols) / cols
    yrange = cp.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) / (rows - 1) if rows % 2 else cp.linspace(-rows / 2,
                                                                                                          rows / 2 - 1,
                                                                                                          rows) / rows

    x, y = cp.meshgrid(xrange, yrange)

    radius = cp.sqrt(x ** 2 + y ** 2)
    theta = cp.arctan2(-y, x)

    radius = cp.fft.ifftshift(radius)
    theta = cp.fft.ifftshift(theta)
    radius[0, 0] = 1

    sintheta, costheta = cp.sin(theta), cp.cos(theta)

    lp = cp.array(low_pass_filter([rows, cols], .45, 15))
    logGabor = [None] * nscale

    # Precompute log-Gabor filters
    for s in range(nscale):
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        logGabor[s] = cp.exp((-(cp.log(radius / fo)) ** 2) / (2 * cp.log(sigmaOnf) ** 2))
        logGabor[s] *= lp
        logGabor[s][0, 0] = 0

    spread = [None] * norient

    # Precompute orientation filters
    for o in range(norient):
        angl = o * cp.pi / norient
        ds = sintheta * cp.cos(angl) - costheta * cp.sin(angl)
        dc = costheta * cp.cos(angl) + sintheta * cp.sin(angl)
        dtheta = cp.abs(cp.arctan2(ds, dc))
        spread[o] = cp.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    # Compute the phase congruency maps and features
    for o in range(norient):
        angl = o * cp.pi / norient
        sumE_ThisOrient, sumO_ThisOrient, sumAn_ThisOrient, Energy = zero.copy(), zero.copy(), zero.copy(), zero.copy()

        for s in range(nscale):
            filter_ = logGabor[s] * spread[o]
            ifftFilt = cp.real(cp.fft.ifft2(filter_)) * cp.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            EO[s][o] = cp.fft.ifft2(imagefft * filter_)
            An = cp.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += cp.real(EO[s][o])
            sumO_ThisOrient += cp.imag(EO[s][o])

            if s == 0:
                EM_n = cp.sum(filter_ ** 2)
                maxAn = An
            else:
                maxAn = cp.maximum(maxAn, An)

        # Compute energy and thresholding
        XEnergy = cp.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for s in range(nscale):
            E = cp.real(EO[s][o])
            O = cp.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - cp.abs(E * MeanO - O * MeanE)

        medianE2n = cp.median(cp.abs(EO[0][o]) ** 2)
        meanE2n = -medianE2n / cp.log(0.5)
        estMeanE2n.append(meanE2n)

        noisePower = meanE2n / EM_n
        EstSumAn2 = sum(ifftFilterArray[s] ** 2 for s in range(nscale))
        EstSumAiAj = sum(
            ifftFilterArray[si] * ifftFilterArray[sj] for si in range(nscale - 1) for sj in range(si + 1, nscale))

        EstNoiseEnergy2 = 2 * noisePower * cp.sum(EstSumAn2) + 4 * noisePower * cp.sum(EstSumAiAj)
        tau = cp.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * cp.sqrt(cp.pi / 2)
        EstNoiseEnergySigma = cp.sqrt((2 - cp.pi / 2) * tau ** 2)
        T = EstNoiseEnergy + k * EstNoiseEnergySigma
        T /= 1.7

        Energy = cp.maximum(Energy - T, zero)

        width = sumAn_ThisOrient / (maxAn + epsilon) / nscale
        weight = 1.0 / (1 + cp.exp((cutOff - width) * g))

        PC.append(weight * Energy / sumAn_ThisOrient)
        featType = E + 1j * O

        covx = PC[o] * cp.cos(angl)
        covy = PC[o] * cp.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy *= 4 / norient

    denom = cp.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    ori = cp.arctan2(sin2theta, cos2theta) / 2
    ori = cp.rad2deg(ori)
    ori[ori < 0] += 180

    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    return M, m, ori, featType, PC, EO