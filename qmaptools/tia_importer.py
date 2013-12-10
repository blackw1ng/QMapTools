def getType(dataType):
    if dataType == 1:
        Type = "B" #uint8
    elif dataType == 2:
        Type = "H" #uint16
    elif dataType == 3:
        Type = "I" #uint32
    elif dataType == 4:
        Type = "b"
    elif dataType == 5:
        Type = "h" #int16
    elif dataType ==  6:
        Type = "i" #int32
    elif dataType ==  7:
        Type = "f" #float32
    elif dataType ==  8:
        Type = "d" #float64
    else:
        print('Unsupported data type') #complex data types are unsupported currently
    return Type
#END getType()

def serReader(fname):

    import numpy as np
    import sys, math
    import array
    
    f = open(fname)
    
    #Setup a some arrays for reading different data types in the header
    readINT16 = array.array("h")
    readINT32 = array.array("i")
    readDOUBLE = array.array("d")
    readCHAR = array.array("b") #unsigned char
    toString = array.array("u") #for converting CHARs to useful strings

    #0 byteorder, 1 seriesID, 2 seriesVersion
    readINT16.fromfile(f,3) #read 3 values

    #0 datatypeID, 1 tagtypeID, 2 totalNumberElements
    #3 validNumberElements, 4 offsetArrayOffset, 5 numberDimensions
    readINT32.fromfile(f,6)
    
    for kk in range(0,readINT32[5]):
        
        readINT32.fromfile(f,1)
            
        readDOUBLE.fromfile(f,2) #0 calibrationOffset, 1 calibrationDelta
        
        readINT32.fromfile(f,2) #7 calibrationElement, 8 descriptionLength
        
        #Describes the dimension (e.g. Number)
        readCHAR.fromfile(f, readINT32[8])
        
        #9 unitsLength
        readINT32.fromfile(f,1)
        
        #unit string (e.g. meters)
        readUnit = array.array("b")
        readUnit.fromfile(f,readINT32[9])
        unit = readUnit.tostring() #the units string (e.g. meters)
        
    #Get arrays containing byte offsets for data and tags
    #Data offset array (the byte offsets of the individual data elements)
    dataOffsetArray = array.array("i")
    dataOffsetArray.fromfile(f,readINT32[2])
    tagOffsetArray = array.array("i")
    tagOffsetArray.fromfile(f,readINT32[2])
    
    #1D data elements datatypeID is hex: 0x4120 or decimal 16672
    if readINT32[0] == 16672:
        print("1D dataset - not tested")
        for ii in range(0,readINT32[6]):
            f.seek(dataOffsetArray[ii],0) #seek to the correct byte offset
            calibration = array.array("d")
            calibrationElement = array.array("i")
            dataType = array.array("h")
            
            #read in the data calibration information
            calibration.fromfile(f,2) # 0 Offset, 1 Delta
            calibrationElement.fromfile(f,1)
            dataType.fromfile(f,1)
            
            Type = getType(dataType[0]) #convert SER datatype to Python datatype string
            arrayLength = array.array("i")
            arrayLength.fromfile(f,1) #get the size of the array
            dataValues = array.array(Type)
            dataValues.fromfile(f,arrayLength[0]) #read in the data
            
            #Convert into numpy array
            datavalues2 = dataValues.tolist()
            dataValuesNP = np.array(dataValues)
            
            #Create spectra array on first pass
            if ii == 0:
            	spectra = np.zeros((readINT32[6]+1,arrayLength[0]))
            
            spectra[ii+1,:] = dataValuesNP
            
        #Add calibrated data to start of spectra array
        bins = np.linspace(calibration[0],calibration[0] + (arrayLength[0]-1)*calibration[1],arrayLength[0])
        spectra[0,:] = bins
		
        print "Offset = %f, Delta = %f, ArraySize = %f" % (calibration[0], calibration[1], arrayLength[0])
		
        return (spectra,["Spectrum",calibration[0], calibration[1]])
            
    #2D data elemets have datatypeID as hex: 0x4122 or decimal 16674
    if readINT32[0] == 16674:
        print("2D dataset")
        for jj in range(0,readINT32[6]):
            f.seek(dataOffsetArray[jj],0) #seek to start of data
            
            #Read calibration data for X
            calibrationX = array.array("d")
            calibrationX.fromfile(f,2) #reads the offset calibrationX[0] and the pixel size calibrationX[1] 
            calibrationElementX = array.array("i")
            calibrationElementX.fromfile(f,1) #reads one value
            
            #Read calibration data for Y
            calibrationY = array.array("d")
            calibrationY.fromfile(f,2) #read two values
            calibrationElementY = array.array("i")
            calibrationElementY.fromfile(f,1)
            
            #read array data type
            dataType = array.array("h")
            dataType.fromfile(f,1)
            Type = getType(dataType[0])
            
            #Get array size values
            arraySize = array.array("i")
            arraySize.fromfile(f,2)
            arraySizeX = arraySize[0]
            arraySizeY = arraySize[1]
            
            #Read data from file into the array
            dataValues = array.array(Type)
            dataValues.fromfile(f,arraySize[0]*arraySize[1])
            dataValues2 = dataValues.tolist()

            #reshape the data into an image with the correct shape
            dataValues = np.array(dataValues)
            dataValues = dataValues.reshape(arraySizeX,arraySizeY)
            
            #Save the pixel values
            pixelSizeX = calibrationX[1] * 1e9
            pixelSizeY = calibrationY[1] * 1e9
            
            print 'Pixel size X (nm) = %f' % pixelSizeX
            print 'Pixel Size Y (nm) = %f' % pixelSizeY
            
            return (dataValues,["2D",pixelSizeX,pixelSizeY]
            
    f.close()