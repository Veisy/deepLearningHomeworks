
import os
import xlsxwriter as xlsxwriter
import time

from train_model import train_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

########################################################################################################################
# initialization

#TODO: We trained D1namo dataset with 1 parameter,
# we then converted values from mmol/L to mg/dL,
# Afterwards we used RMSE, MAE, MAPE, R2 scores to evaluate the model

seedRange       =   1                 # seed number will be changed till 'seedRange'
epochRunning    =   100                  # epoch number for running
featureNumber   =   1                    # 1 --> CGM only, 2 --> CGM + Basal Insulin, 3 --> CGM + Basal Insulin + CHO
testFlag        =   1                    # if test flag is 1, test code will run. If it is 0, it will not run.
plotFlag        =   0                    # if plot flag is 1, plots will appear. If it is 1, plots will not appear.
patientNumber   =   9                  # total number of patients
start           =   time.time()          # record start time
seedList        =   list(range(0, seedRange))


modelList   = [ "RNN", "LSTM", "GRU", "BiRNN", "BiLSTM", "BiGRU", "ConvRNN",
              "ConvLSTM", "ConvGRU"   ]

wsList      = [  'patient1', 'patient2', 'patient3', 'patient4', 'patient5', 'patient6','patient7','patient8','patient9']

attentionList = ["no_Attention", "with_Attention"]


########################################################################################################################

#  Running simulation

for modelType, model in enumerate(modelList):

    for attentionFlag, attention in enumerate(attentionList):

        for layerNumber in 1, 2:

            workbook = xlsxwriter.Workbook(
                model + "_layer_" + f"{layerNumber}" + "_" + attention + ".xlsx")
            rmseValList = []
            rmseTestList = []
            maeList = []
            mapeList = []
            rList = []
            worksheet = workbook.add_worksheet(model)
            worksheet.write('A1', 'Patient Number')
            worksheet.write('B1', 'Val RMSE')
            worksheet.write('C1', 'MAE')
            worksheet.write('D1', 'MAPE')
            worksheet.write('E1', 'R2 SCORE')

            for patientFlag in range(patientNumber):
                print(f"Patient {wsList[patientFlag]} is in progress")
                rmseVal, rmseTest, mae, mape_er, r = train_model(1, epochRunning, modelType, testFlag, patientFlag, layerNumber,
                                                                 featureNumber, plotFlag, attentionFlag)
                rmseValList.append(rmseVal)
                rmseTestList.append(rmseTest)
                maeList.append(mae)
                mapeList.append(mape_er)
                rList.append(r)

            worksheet.write_column(1, 0, wsList)
            worksheet.write_column(1, 1, rmseValList)
            worksheet.write_column(1, 2, maeList)
            worksheet.write_column(1, 3, mapeList)
            worksheet.write_column(1, 4, rList)
            workbook.close()
            end = time.time()
            print("The time of execution of above program is :",
                  (end - start) / 60, "m")



