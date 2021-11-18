import time
import sys
import pytz
import logging
import datetime
from urllib.parse import urlparse
import schedule
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from opcua import ua, Client
import numpy as np

import scipy.signal
import math
from dateutil import parser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import email.mime.application
import email
import mimetypes


class HighPassFilter(object):

    @staticmethod
    def get_highpass_coefficients(lowcut, sampleRate, order=5):
        nyq = 0.5 * sampleRate
        low = lowcut / nyq
        b, a = scipy.signal.butter(order, [low], btype='highpass')
        return b, a

    @staticmethod
    def run_highpass_filter(data, lowcut, sampleRate, order=5):
        if lowcut >= sampleRate/2.0:
            return data*0.0
        b, a = HighPassFilter.get_highpass_coefficients(lowcut, sampleRate, order=order)
        y = scipy.signal.filtfilt(b, a, data, padtype='even')
        return y
    
    @staticmethod
    def perform_hpf_filtering(data, sampleRate, hpf=3):
        if hpf == 0:
            return data
        data[0:6] = data[13:7:-1] # skip compressor settling
        data = HighPassFilter.run_highpass_filter(
            data=data,
            lowcut=3,
            sampleRate=sampleRate,
            order=1,
        )
        data = HighPassFilter.run_highpass_filter(
            data=data,
            lowcut=int(hpf),
            sampleRate=sampleRate,
            order=2,
        )
        return data

class FourierTransform(object):

    @staticmethod
    def perform_fft_windowed(signal, fs, winSize, nOverlap, window, detrend = True, mode = 'lin'):
        assert(nOverlap < winSize)
        assert(mode in ('magnitudeRMS', 'magnitudePeak', 'lin', 'log'))
    
        # Compose window and calculate 'coherent gain scale factor'
        w = scipy.signal.get_window(window, winSize)
        # http://www.bores.com/courses/advanced/windows/files/windows.pdf
        # Bores signal processing: "FFT window functions: Limits on FFT analysis"
        # F. J. Harris, "On the use of windows for harmonic analysis with the
        # discrete Fourier transform," in Proceedings of the IEEE, vol. 66, no. 1,
        # pp. 51-83, Jan. 1978.
        coherentGainScaleFactor = np.sum(w)/winSize
    
        # Zero-pad signal if smaller than window
        padding = len(w) - len(signal)
        if padding > 0:
            signal = np.pad(signal, (0,padding), 'constant')
    
        # Number of windows
        k = int(np.fix((len(signal)-nOverlap)/(len(w)-nOverlap)))
    
        # Calculate psd
        j = 0
        spec = np.zeros(len(w));
        for i in range(0, k):
            segment = signal[j:j+len(w)]
            if detrend is True:
                segment = scipy.signal.detrend(segment)
            winData = segment*w
            # Calculate FFT, divide by sqrt(N) for power conservation,
            # and another sqrt(N) for RMS amplitude spectrum.
            fftData = np.fft.fft(winData, len(w))/len(w)
            sqAbsFFT = abs(fftData/coherentGainScaleFactor)**2
            spec = spec + sqAbsFFT;
            j = j + len(w) - nOverlap
    
        # Scale for number of windows
        spec = spec/k
    
        # If signal is not complex, select first half
        if len(np.where(np.iscomplex(signal))[0]) == 0:
            stop = int(math.ceil(len(w)/2.0))
            # Multiply by 2, except for DC and fmax. It is asserted that N is even.
            spec[1:stop-1] = 2*spec[1:stop-1]
        else:
            stop = len(w)
        spec = spec[0:stop]
        freq = np.round(float(fs)/len(w)*np.arange(0, stop), 2)
    
        if mode == 'lin': # Linear Power spectrum
            return (spec, freq)
        elif mode == 'log': # Log Power spectrum
            return (10.*np.log10(spec), freq)
        elif mode == 'magnitudeRMS': # RMS Magnitude spectrum
            return (np.sqrt(spec), freq)
        elif mode == 'magnitudePeak': # Peak Magnitude spectrum
            return (np.sqrt(2.*spec), freq)
class OpcUaClient(object):
    CONNECT_TIMEOUT = 15  # [sec]
    RETRY_DELAY = 10  # [sec]
    MAX_RETRIES = 3  # [-]

    class Decorators(object):
        @staticmethod
        def autoConnectingClient(wrappedMethod):
            def wrapper(obj, *args, **kwargs):
                for retry in range(OpcUaClient.MAX_RETRIES):
                    try:
                        return wrappedMethod(obj, *args, **kwargs)
                    except ua.uaerrors.BadNoMatch:
                        raise
                    except Exception:
                        pass
                    try:
                        obj._logger.warning('(Re)connecting to OPC-UA service.')
                        obj.reconnect()
                    except ConnectionRefusedError:
                        obj._logger.warning(
                            'Connection refused. Retry in 10s.'.format(
                                OpcUaClient.RETRY_DELAY
                            )
                        )
                        time.sleep(OpcUaClient.RETRY_DELAY)
                else:  # So the exception is exposed.
                    obj.reconnect()
                    return wrappedMethod(obj, *args, **kwargs)
            return wrapper

    def __init__(self, serverUrl):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = Client(
            serverUrl.geturl(),
            timeout=self.CONNECT_TIMEOUT
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        self._client = None

    @property
    @Decorators.autoConnectingClient
    def sensorList(self):
        return self.objectsNode.get_children()

    @property
    @Decorators.autoConnectingClient
    def objectsNode(self):
        path = [ua.QualifiedName(name='Objects', namespaceidx=0)]
        return self._client.get_root_node().get_child(path)

    def connect(self):
        self._client.connect()
        self._client.load_type_definitions()

    def disconnect(self):
        try:
            self._client.disconnect()
        except Exception:
            pass

    def reconnect(self):
        self.disconnect()
        self.connect()

    @Decorators.autoConnectingClient
    def get_browse_name(self, uaNode):
        return uaNode.get_browse_name()

    @Decorators.autoConnectingClient
    def get_node_class(self, uaNode):
        return uaNode.get_node_class()

    @Decorators.autoConnectingClient
    def get_namespace_index(self, uri):
        return self._client.get_namespace_index(uri)

    @Decorators.autoConnectingClient
    def get_child(self, uaNode, path):
        return uaNode.get_child(path)

    @Decorators.autoConnectingClient
    def read_raw_history(self,
                         uaNode,
                         starttime=None,
                         endtime=None,
                         numvalues=0,
                         cont=None):
        details = ua.ReadRawModifiedDetails()
        details.IsReadModified = False
        details.StartTime = starttime or ua.get_win_epoch()
        details.EndTime = endtime or ua.get_win_epoch()
        details.NumValuesPerNode = numvalues
        details.ReturnBounds = True
        result = OpcUaClient._history_read(uaNode, details, cont)
        assert(result.StatusCode.is_good())
        return result.HistoryData.DataValues, result.ContinuationPoint

    @staticmethod
    def _history_read(uaNode, details, cont):
        valueid = ua.HistoryReadValueId()
        valueid.NodeId = uaNode.nodeid
        valueid.IndexRange = ''
        valueid.ContinuationPoint = cont

        params = ua.HistoryReadParameters()
        params.HistoryReadDetails = details
        params.TimestampsToReturn = ua.TimestampsToReturn.Both
        params.ReleaseContinuationPoints = False
        params.NodesToRead.append(valueid)
        result = uaNode.server.history_read(params)[0]
        return result


class DataAcquisition(object):
    LOGGER = logging.getLogger('DataAcquisition')
    MAX_VALUES_PER_ENDNODE = 10000  # Num values per endnode
    MAX_VALUES_PER_REQUEST = 10  # Num values per history request

    @staticmethod
    def get_sensor_sub_node(client, macId, browseName, subBrowseName, sub2BrowseName=None, sub3BrowseName=None, sub4BrowseName=None):
        nsIdx = client.get_namespace_index(
            'http://www.iqunet.com'
        )  # iQunet namespace index
        bpath = [
            ua.QualifiedName(name=macId, namespaceidx=nsIdx),
            ua.QualifiedName(name=browseName, namespaceidx=nsIdx),
            ua.QualifiedName(name=subBrowseName, namespaceidx=nsIdx)
        ]
        if sub2BrowseName is not None:
            bpath.append(ua.QualifiedName(name=sub2BrowseName, namespaceidx=nsIdx))
        if sub3BrowseName is not None:
            bpath.append(ua.QualifiedName(name=sub3BrowseName, namespaceidx=nsIdx))
        if sub4BrowseName is not None:
            bpath.append(ua.QualifiedName(name=sub4BrowseName, namespaceidx=nsIdx))
        sensorNode = client.objectsNode.get_child(bpath)
        return sensorNode

    @staticmethod
    def get_endnode_data(client, endNode, starttime, endtime):
        dvList = DataAcquisition.download_endnode(
            client=client,
            endNode=endNode,
            starttime=starttime,
            endtime=endtime
        )
        dates, values = ([], [])
        for dv in dvList:
            dates.append(dv.SourceTimestamp.strftime('%Y-%m-%d %H:%M:%S'))
            values.append(dv.Value.Value)

        # If no starttime is given, results of read_raw_history are reversed.
        if starttime is None:
            values.reverse()
            dates.reverse()
        return (values, dates)

    @staticmethod
    def download_endnode(client, endNode, starttime, endtime):
        endNodeName = client.get_browse_name(endNode).Name
        DataAcquisition.LOGGER.info(
            'Downloading endnode {:s}'.format(
                endNodeName
            )
        )
        dvList, contId = [], None
        while True:
            remaining = DataAcquisition.MAX_VALUES_PER_ENDNODE - len(dvList)
            assert(remaining >= 0)
            numvalues = min(DataAcquisition.MAX_VALUES_PER_REQUEST, remaining)
            partial, contId = client.read_raw_history(
                uaNode=endNode,
                starttime=starttime,
                endtime=endtime,
                numvalues=numvalues,
                cont=contId
            )
            if not len(partial):
                DataAcquisition.LOGGER.warning(
                    'No data was returned for {:s}'.format(endNodeName)
                )
                break
            dvList.extend(partial)
            sys.stdout.write('\r    Loaded {:d} values, {:s} -> {:s}'.format(
                len(dvList),
                str(dvList[0].ServerTimestamp.strftime("%Y-%m-%d %H:%M:%S")),
                str(dvList[-1].ServerTimestamp.strftime("%Y-%m-%d %H:%M:%S"))
            ))
            sys.stdout.flush()
            if contId is None:
                break  # No more data.
            if len(dvList) >= DataAcquisition.MAX_VALUES_PER_ENDNODE:
                break  # Too much data.
        sys.stdout.write('...OK.\n')
        return dvList

    @staticmethod
    def get_anomaly_model_nodes(client, macId):
        sensorNode = \
            DataAcquisition.get_sensor_sub_node(client, macId, "tensorFlow", "models")
        DataAcquisition.LOGGER.info(
            'Browsing for models of {:s}'.format(macId)
        )
        modelNodes = sensorNode.get_children()
        return modelNodes

    @staticmethod
    def get_anomaly_model_parameters(client, macId, starttime, endtime):
        #acquires a list of all subnodes below the models node
        modelNodes = \
            DataAcquisition.get_anomaly_model_nodes(client, macId)
        #to here
        models = dict()
        for mnode in modelNodes:
            key = mnode.get_display_name().Text
            print(key)
            sensorNode = \
                DataAcquisition.get_sensor_sub_node(client, macId, "tensorFlow", "models", key, "lossMAE")
            (valuesraw, datesraw) = \
                DataAcquisition.get_endnode_data(
                    client=client,
                    endNode=sensorNode,
                    starttime=starttime,
                    endtime=endtime
            )

            sensorNode = \
                DataAcquisition.get_sensor_sub_node(client, macId, "tensorFlow", "models", key, "lossMAE", "alarmLevel")
            alarmLevel = sensorNode.get_value()
            modelSet = {
                "raw": (valuesraw, datesraw),
                "alarmLevel": alarmLevel
            }
            models[key] = modelSet
        return models
    @staticmethod
    def get_sensor_data(serverUrl, macId, browseName, starttime, endtime):
        with OpcUaClient(serverUrl) as client:
            assert(client._client.uaclient._uasocket.timeout == 15)
            sensorNode = \
                DataAcquisition.get_sensor_node(client, macId, browseName)
            DataAcquisition.LOGGER.info(
                    'Browsing {:s}'.format(macId)
            )
            (values, dates) = \
                DataAcquisition.get_endnode_data(
                        client=client,
                        endNode=sensorNode,
                        starttime=starttime,
                        endtime=endtime
                )
        return (values, dates)
    @staticmethod
    def get_sensor_node(client, macId, browseName):
        nsIdx = client.get_namespace_index(
                'http://www.iqunet.com'
        )  # iQunet namespace index
        bpath = [
                ua.QualifiedName(name=macId, namespaceidx=nsIdx),
                ua.QualifiedName(name=browseName, namespaceidx=nsIdx)
        ]
        sensorNode = client.objectsNode.get_child(bpath)
        return sensorNode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("opcua").setLevel(logging.WARNING)

    # replace xx.xx.xx.xx with the IP address of your server
    serverIP = "25.100.199.132"
    serverUrl = urlparse('opc.tcp://{:s}:4840'.format(serverIP))

    # replace xx:xx:xx:xx with your sensors macId
    macId = 'e0:10:9a:cd'
    sensorName = 'Kowon_Vib_BackCrank'

    rangeTime = 20
    # startTime = datetime.datetime.now()
    # startTime = datetime.datetime(2021,2,16,0,0,0)
    startTime = datetime.datetime(2021,5,16,15,0,0)

    timeZone = "Asia/Seoul"
    hpf = 3

    def plotData():
        global startTime
        global endTime
        global hpf
        # acquire history data
        browseName=["accelerationPack","axis","batteryVoltage","boardTemperature",
                   "firmware","formatRange","gKurtX","gKurtY","gKurtZ","gRmsX","gRmsY",
                   "gRmsZ","hardware","mmsKurtX","mmsKurtY","mmsKurtZ",
                   "mmsRmsX","mmsRmsY","mmsRmsZ","numSamples","sampleRate"]
    
        (values,dates) = DataAcquisition.get_sensor_data(
            serverUrl=serverUrl,
            macId=macId,
            browseName=browseName[0],
            starttime=startTime,
            endtime=endTime
            )
        
      
        # convert vibration data to 'g' units and plot data
        data = [val[1:-6] for val in values]
        numSamples = [val[0] for val in values]
        sampleRates = [val[-6] for val in values]    
        fRanges = [val[-5] for val in values]
        axes = [val[-3] for val in values]
        axis='XYZ'    
        
        for i in range(len(fRanges)):
            data[i] = [d/512.0*fRanges[i] for d in data[i]]
            maxTimeValue = numSamples[i]/sampleRates[i]
            stepSize = 1/sampleRates[i]
            timeValues = np.arange(0, maxTimeValue, stepSize)
            
            data[i] = HighPassFilter.perform_hpf_filtering(
            data=data[i],
            sampleRate=sampleRates[i], 
            hpf=hpf
            )
            
    
            windowSize = len(data[i]) # window size
            nOverlap   = 0 # overlap window
            windowType = 'hann' # hanning window     
            mode       = 'magnitudeRMS' # RMS magnitude spectrum.
            (npFFT, npFreqs) = FourierTransform.perform_fft_windowed(
                signal=values[i], 
                fs=sampleRates[i],
                winSize=windowSize,
                nOverlap=nOverlap, 
                window=windowType, 
                detrend = False, 
                mode = mode)
    
            # Write to csv files       
            
            # comment out unnessesary code for efficiency except local time sunwook
            # tdata = { "Time [s]": timeValues, 'RMS Acceleration [g]' : data[i] }
            # fdata = { "Frequency [Hz]": npFreqs, 'RMS Acceleration [g]' : npFFT}
            # df = pd.DataFrame(tdata)
            # df2= pd.DataFrame(fdata)
            local_time=parser.parse(dates[i]).astimezone(pytz.timezone(timeZone))
            
            #comment out csv sunwook
            # FN = "sensor_time_"+ axis +"-" + str(local_time.strftime("%Y %b %d T %H:%M:%S")).replace(':','-') + '.' + str(i) + ".csv"
            # FN2 = "sensor_freq_"+ axis +"-" + str(local_time.strftime("%Y %b %d T %H:%M:%S")).replace(':','-') + '.' + str(i) + ".csv"
     #       FN = "sensor_time_"+ axis[i] +"-" + str(local_time.strftime("%Y %b %d T %H:%M:%S")).replace(':','-') + '.' + str(i) + ".csv"
     #       FN2 = "sensor_freq_"+ axis[i] +"-" + str(local_time.strftime("%Y %b %d T %H:%M:%S")).replace(':','-') + '.' + str(i) + ".csv"
    
    
            #df.to_excel(FN, sheet_name='sheet1')
            # comment out csv save sunwook
            # df.to_csv(FN)
            # df2.to_csv(FN2)        
    
            # Plot Figures, time domain
            
            plt.figure(); plt.subplot(2,1,1); plt.plot(timeValues, data[i])
            title = (local_time + datetime.timedelta(seconds=.5)).replace(microsecond=0)
            filetitle = title.strftime("%Y-%m-%d %H-%M-%S")

            title = title.strftime("%a %b %d %Y %H:%M:%S")+" "+axis +" axis"
            plt.title(title)

            plt.grid(True)
            plt.xlabel('Time [s]')
            plt.ylabel('RMS Acceleration [g]')
    
    
            #plot frequency domain
            plt.subplot(2,1,2);plt.plot(npFreqs, npFFT)
            plt.xlim((0, sampleRates[i]/2)) 
            # plt.xlim(-0.02, 0.05)

            viewPortOptions = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
            viewPort = [i for i in viewPortOptions if i >= max(npFFT)][0]
            plt.ylim((0,viewPort))
            # plt.ylim(-0.04, 0.04)

            plt.grid(True)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('RMS Acceleration [g]')     
            
            # save RMS acceleration plot as image sunwook
            plt.savefig("RMS acceleration plot "+ filetitle + ".png")
    
    def send_email():
        gmail_user = 'reshenie.work@gmail.com'
        gmail_password = 'reshenie12!'
        text_type = 'plain' # or 'html'
        text = """\
            - 센서 이름 : %s
            - 이상 징후 감지 센서 : %s
            - 이상 징후 발생 일시 : %s
            
            자세한 데이터는 Dashboard를 참고해주시기 바랍니다.""" % (sensorName,str(model), str(rawdates[i]))
        msg = MIMEMultipart('mixed')
        msg['Subject'] = '[예지보전 시스템] *** 설비 이상 징후가 감지되었습니다.'
        msg['From'] = gmail_user
        #msg['To'] = 'reshenie.work+testemail@gmail.com'

        recipients = ['kslee@kowonmetal.com', 'jhhuh@kowonmetal.com', 'wwchoi@kowonmetal.com', 'bdna@kowonmetal.com', 'reshenie.work@gmail.com']
        msg['To'] = (', ').join(recipients)

        msg.attach(MIMEText(text, text_type, 'utf-8'))
        
        
        rawDate = str(rawdates[i])
        newDate = rawDate.replace(":", "-")
        print(newDate)
        filename = "RMS acceleration plot "+ newDate + ".png"
        fp=open(filename,'rb')
        att = email.mime.application.MIMEApplication(fp.read(),_subtype="png")
        fp.close()
        att.add_header('Content-Disposition','attachment',filename=filename)
        msg.attach(att)
        
        
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        # or server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()

    def grab_anomaly():
        global startTime
        global endTime
        global rawdates
        global raw
        global i
        global model
        print("Running script")
        print(startTime)
        endTime = startTime + datetime.timedelta(minutes=rangeTime)
        print(endTime)

        starttime = startTime.strftime('%Y-%m-%d %H:%M:%S')
        endtime = endTime.strftime('%Y-%m-%d %H:%M:%S')

        # format start and end time
        starttime = pytz.utc.localize(
            datetime.datetime.strptime(starttime, '%Y-%m-%d %H:%M:%S')
        )
        endtime = pytz.utc.localize(
            datetime.datetime.strptime(endtime, '%Y-%m-%d %H:%M:%S')
        )

        # create opc ua client
        with OpcUaClient(serverUrl) as client:
            assert(client._client.uaclient._uasocket.timeout == 15)

            # acquire model data
            modelDict = DataAcquisition.get_anomaly_model_parameters(
                client=client,
                macId=macId,
                starttime=starttime,
                endtime=endtime
            )
            for model in modelDict.keys():
                # plt.figure()
                raw = modelDict[model]["raw"][0]
                rawdates = modelDict[model]["raw"][1]
                alarmLevel = modelDict[model]["alarmLevel"]
                for i in range(len(raw)):

                    print(raw[i])
                    print('For axis %s' % (str(model)))
                    if alarmLevel < raw[i]:
                        plotData()
                        print('WARNING WARNING WARNING WARNING anomaly found at %s' % (str(rawdates[i])))
                        send_email()
                    else:
                        print('Large value found but is not an anomaly at %s' % (str(rawdates[i])))
                        
        startTime = endTime

    schedule.every(20).minutes.do(grab_anomaly)

    while True:
        schedule.run_pending()
        time.sleep(1)
