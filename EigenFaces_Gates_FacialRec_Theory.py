# ANLY503Eigenfaces.py
# Gates
# https://pypi.python.org/pypi/Pillow/3.3.1
# Interesting Reference
# http://www.pythonware.com/media/data/pil-handbook.pdf

import numpy as np
from PIL import Image


def main():
    path = "./FacesDataFullSizeDataset"  ##40 folders each with 10 images
    # Transform Image data into a Faces Matrix
    Face_Matrix, original_shape, num_images = ReadInFaceDataToMatrix(path)
    print("The Face Matrix is \n", Face_Matrix)
    print("The shape of the Face Matrix is\n", Face_Matrix.shape)
    # Normalize the Faces Matrix
    Norm_Face_Matrix, Mean_Vector = NormalizeTheMatrix(Face_Matrix, num_images, original_shape)
    # Perform PCA on the normalized Faces Matrix to get the
    # Eigenvectors (eigen faces) and eigenvalues
    # k is the number of eigen faces (eigenvectors) we want
    k = 20
    TopEigFacesMatrix = GetEigenFaces(Norm_Face_Matrix, num_images, k, original_shape, Mean_Vector)
    # Given a face - match it to the dataset
    # @@@
    MatchFace(k,TopEigFacesMatrix,Mean_Vector,original_shape,num_images,Norm_Face_Matrix)


def ReadInFaceDataToMatrix(p):
    # Assume all images are the same shape
    # Get the shape once
    # !! UPDATE THIS PART DEPENDING ON THE DATA PATH etc.
    # fullpath=p+"\Face1.pgm"

    partsall = '/s' + str(1) + "/" + str(1) + ".pgm"
    # print(partsall)
    fullpath = p + partsall
    # print(fullpath)
    # img1=Image.open(fullpath).convert('RGBA') #For color images

    ##Get the very first image
    # **************
    img1 = Image.open(fullpath).convert('L')
    imagearray1 = np.array(img1)
    original_shape = imagearray1.shape
    flat1 = imagearray1.ravel()
    facevector1 = np.matrix(flat1)
    facematrix = facevector1
    # print("The first facevector is ",facevector1)
    # print("The current facematrix is ", facematrix)
    shape = facematrix.shape
    # print("The current facematrix shape is",shape)

    # Load all the images into a matrix column-wise
    # https://www.ibm.com/developerworks/community/
    # blogs/jfp/entry/Elementary_Matrix_Operations_In_Python?lang=en

    ## ------- num_folders=15 ## This is for FACESdata_medium and must be updated per data
    ##---------num_files=10
    num_folders = 40  # 40
    num_files = 10
    file_counter = 0
    for n in range(num_folders):
        # fullpath=p+"\Face"+str(i+2)+".pgm" #For FACESdata_small
        folder = str(n + 1)  # Folder names are s1, s2, etc.
        for file in range(num_files):
            if (folder == "1" and file == 0):
                pass
            else:
                partsall = '/s' + folder + "/" + str(file + 1) + ".pgm"
                # file=str(file+1)+".pgm" #Files are named 1.pgm, 2.pgm, etc.
                fullpath = p + partsall
                # print(fullpath)
                # img=Image.open(fullpath).convert('RGBA') #For color images
                # ************
                img = Image.open(fullpath).convert('L')
                imagearray = np.array(img)
                # make a 1-dimensional view of imagearray
                flat = imagearray.ravel()
                # convert it to a matrix
                facevector = np.matrix(flat)
                # print("The ",file_counter,"facevector is",facevector,"\n")
                facematrix = np.r_[facematrix, facevector]  # row cat
                # print("After counter ",file_counter,"the facematrix is", facematrix,"\n")
                file_counter = file_counter + 1

    # print("The facematrix is\n",facematrix)
    file_counter = file_counter + 1
    # print("\nThe shape of the final facematrix is\n",facematrix.shape)
    # print("The number of files in the dataset is\n",file_counter)

    # print(facematrix)
    # print(facematrix.shape)
    ## TRANSPOSE the Facematrix so that the columns are the faces
    facematrix_t = np.transpose(facematrix)
    # print("The Face Matrix Shape after transpose",facematrix_t.shape)
    # print("The shape is\n",facematrix_t.shape)

    # View all of the data face images
    # reform a numpy array of the original shape
    # print(facematrix[0,:])
    # print(facematrix[0,:].shape)

    #######################################################################
    ## Uncomment this block to view and save all faces in
    ## the dataset
    #######################################################################
    # for i in range(5):
    # face_example = np.asarray(facematrix_t[:,i]).reshape(original_shape)
    #  #print("The facematrix[:,0] is",facematrix_t[:,0], "shape", facematrix_t[:,0].shape)
    #  #print("The face_example is",face_example, "shape", face_example.shape)
    # #print("The ogi shape is", original_shape)
    # # make a PIL image and save it to jpg
    # #face_example_img = Image.fromarray(face_example, 'RGBA')
    # face_example_img = Image.fromarray(face_example, 'L')
    # face_example_img.show()
    # filename="FaceExample"+str(i)+".jpg"
    # face_example_img.save(filename)
    #######################################################################

    return facematrix_t, original_shape, file_counter


def NormalizeTheMatrix(Face_Matrix, num_images, original_shape):
    # Get the mean
    # print("The number of images is\n",num_images,"\n")
    # print(Face_Matrix.shape)
    # To get the mean - sum all columns (since each)
    # image is in a column in this exampe
    # Divide the sum by the number of images

    MeanVector = np.mean(Face_Matrix, axis=1)
    MeanVector = MeanVector.astype(int)
    # MeanVector=np.transpose(MeanVector)

    NormMatrix = Face_Matrix - MeanVector
    # print("The norm matrix is\n",NormMatrix,"\n")

    # print("The mean vector is\n",MeanVector,"\n")
    # print("SHAPE",MeanVector.shape)
    # print("shape of one col of face vec",Face_Matrix[:,0])
    # print("The face matrix column 0 is\n",Face_Matrix[:,0],"\n")

    # VIEW the mean face vector
    meanV_example = np.asarray(MeanVector).reshape(original_shape)
    # make a PIL image and save it to jpg
    # face_example_img = Image.fromarray(face_example, 'RGBA')
    meanV_example_img = Image.fromarray(meanV_example, 'RGBA')
    # meanV_example_img.show()
    filename = "FaceExampleOutputMEANV.jpg"
    # meanV_example_img.save(filename)

    # View the first image read in from the dataset and now the
    # first column of the face matrix.
    meanV_example = np.asarray(Face_Matrix[:, 0]).reshape(original_shape)
    # make a PIL image and save it to jpg
    # face_example_img = Image.fromarray(face_example, 'RGBA')
    meanV_example_img = Image.fromarray(meanV_example, 'L')
    # meanV_example_img.show()
    filename = "FaceExampleOutputFACE0.jpg"
    meanV_example_img.save(filename)

    return NormMatrix, MeanVector


def GetEigenFaces(Norm_Face_Matrix, num_images, k, original_shape, Mean_Vector):
    # Use PCA to determine the eigenvectors and values
    # Get the covariance matrix
    print("Finding the top ", k, "eigenvectors\n")
    # Covariance Matrix (Turk and Pentland) version
    # Is A transpose * A  (rather than AAT)
    Norm_Face_Matrix_t = np.transpose(Norm_Face_Matrix)
    CovMatrix = np.matmul(Norm_Face_Matrix_t, Norm_Face_Matrix)
    # print("The covar matrix is\n",CovMatrix)
    # print("The shape of the Covar matrix is\n",CovMatrix.shape,"\n")

    ##Calculate the eigenvalues and eigenvectors
    evals, evects = np.linalg.eig(CovMatrix)
    # print("The eigenvalues are\n",evals)
    # print("The eigenvectors are\n",evects)
    # Convert the top k eigenvectors back to
    # The orignal space
    # Put the eigenvectors in sorted order
    index = np.argsort(evals)
    # Reverse the result
    index[:] = index[::-1]
    # print(index)
    # Order the eigenvalues from big to small
    evals = evals[index]
    print("The eigenvalues sorted are\n", evals, "\n")
    # Order the eigenvectors per eigenvalues
    evects = evects[:, index]
    # print("The sorted eig vectors are\n",evects,"\n")
    # print("The eigenvalues are\n",evals)
    # print("The eigenvectors are\n",evects)

    # CHoose the top k eigenvectors
    TopEigVs = evects[:k, :]
    TopEigV_t = np.transpose(TopEigVs)
    print("The top eigvects transposed are\n", TopEigV_t)
    print("Top EigenVectors transposed shape", TopEigV_t.shape)

    ## CONVERT back to original space

    Top_k_EigenFaces = np.matmul(Norm_Face_Matrix, TopEigV_t)

    # print("The top k eigfaces matrix is\n" ,Top_k_EigenFaces)
    # print("The shape of the TopEigfaces is", Top_k_EigenFaces.shape)
    # print("The shape of norm face matrix is",Norm_Face_Matrix.shape)

    ##View the Eigenfaces
    for i in range(k):
        # print(TopEigVs[i,:])    ((r*c,N) * (N,1)--> (r*c, 1))
        V = np.matmul(Norm_Face_Matrix, TopEigV_t[:, i])

        V = V.astype(int)

        # View each eigenvector
        face_example = np.asarray(V).reshape(original_shape)

        # make a PIL image and save it to jpg

        face_example_img = Image.fromarray(face_example, 'RGBA')
        # face_example_img.show()

        filename = "FaceExampleOutput" + str(i + 1) + ".jpg"

        # face_example_img.save(filename) ## ISSUE WITH THIS LINE**

        # print(V)

    return Top_k_EigenFaces


def MatchFace(k, TopEigFacesMatrix, Mean_Vector, original_shape, num_images, Norm_Face_Matrix):
    # print("The TopEigFacesMatrix is\n",TopEigFacesMatrix,"\n")

    #### Read in and convert a face to be matched
    TestFaceImage = "6_wasin21.pgm"
    img = Image.open(TestFaceImage).convert('L')
    print("here1")
    imagearray = np.array(img)
    flat = imagearray.ravel()
    facevector = np.matrix(flat)
    testfacevector = np.transpose(facevector)
    testshape = testfacevector.shape  ## create a column vector
    # print("The test face matrix is\n",testfacevector)
    # print("\nThe shape is\n",testshape)

    ####Subtract the mean from the test face
    # print("The test face matrix is\n",testfacevector)
    # print("The mean matrix is\n",Mean_Vector)
    norm_test_face = testfacevector - Mean_Vector
    # print("The norm test V is\n",norm_test_face)

    ### Project the norm test face onto the eigenfaces space
    # Using the eigenfaces
    ##Calculate the weight of each eigenface as it compares to the
    ## test face vector

    EigenTestFaceProjection = np.matmul(np.transpose(TopEigFacesMatrix), norm_test_face)
    # print("The projection matrix is\n",EigenTestFaceProjection)

    ##The DISTANCE between EigenFaceProjection and all other faces-mean in the dataset
    ProjDataFace = np.matmul(np.transpose(TopEigFacesMatrix), Norm_Face_Matrix[:, 0])
    print("The shape of the first vector (face) in the norm face matrix projected into Eig Space is: ",
          ProjDataFace.shape)
    # print("The first projected data face is",ProjDataFace)
    distance = np.sqrt(np.sum((np.square(ProjDataFace - EigenTestFaceProjection)), axis=0))
    distance = int(np.ndarray.item(distance))
    distance1 = distance
    min_database_face = Norm_Face_Matrix[:, 0]

    print("The shape of Norm_Face_Matrix is: ", Norm_Face_Matrix.shape)
    print("The shape of the Eigenfaces Matrix is: ", TopEigFacesMatrix.shape)

    for i in range(num_images - 1):
        # print(Norm_Face_Matrix[:,i])
        ##Measure the euclidean dist between test face and all other dataset
        ##project the dataset face into eigenface space
        ProjDataFace = np.matmul(np.transpose(TopEigFacesMatrix), Norm_Face_Matrix[:, i + 1])
        # print("Projected data base face is\n",ProjDataFace)
        # print("Projected test face is\n",EigenTestFaceProjection)
        ##Find distance between database face and testface
        # print("square diff", np.square(ProjDataFace - EigenTestFaceProjection))
        # print("sum squ diff is axis 0",np.sum((np.square(ProjDataFace - EigenTestFaceProjection)),axis=0))
        # print("ROOT",np.sqrt(np.sum((np.square(ProjDataFace - EigenTestFaceProjection)),axis=0)))
        newdistance = np.sqrt(np.sum((np.square(ProjDataFace - EigenTestFaceProjection)), axis=0))
        newdistance = int(np.ndarray.item(newdistance))
        # print("New dist is",newdistance)
        # print("The distance is",distance)
        # print('Distance and newdistance',distance,"and",newdistance )
        if (distance > newdistance):
            distance = newdistance
            min_database_face = Norm_Face_Matrix[:, i + 1]

    # print("Min dist",distance)
    # print("Min face", min_database_face)
    # print("distance1 is",distance1)

    ##Render the predicted face
    test_prediction = np.asarray(min_database_face + Mean_Vector).reshape(original_shape)
    # make a PIL image and save it to jpg
    # face_example_img = Image.fromarray(face_example, 'RGBA')
    test_example_img = Image.fromarray(test_prediction, 'L')
    test_example_img.show()
    # test_example_img = test_example_img.convert('RGB')
    filename = "FaceExampleTestPrediction.jpg"
    test_example_img.save(filename)

    ##Render the actual test face
    test_prediction = np.asarray(testfacevector).reshape(original_shape)
    # make a PIL image and save it to jpg
    # face_example_img = Image.fromarray(face_example, 'RGBA')
    # ***************
    test_example_img = Image.fromarray(test_prediction, 'L')
    test_example_img.show()
    test_example_img = test_example_img.convert('RGB')
    filename = "FaceExampleTestOriginal.jpg"
    test_example_img.save(filename)

main()
