# Joe Hollowed
# University of Michigan 2023
# 
# This class provides a graphical user interface for inspecting zonally-averaged variables for user-defined latitude bands in CLDERA HSW++ datasets

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import numpy as np
from data_handler import data_handler
from data_downloader import download_data


# ==================================================================


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        '''
        This function generated by QtDesigner; builds the main elements of the GUI
        '''
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1540, 695)
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        
        # set some global attributes
        
        # -------------------------------------------------------
        # -------------------- options panel --------------------

        self.optionsPanel = QtWidgets.QGroupBox(self.centralwidget)
        self.optionsPanel.setGeometry(QtCore.QRect(10, 0, 781, 281))
        self.optionsPanel.setCheckable(False)
        self.optionsPanel.setObjectName("optionsPanel")
        
        self.dataReleaseText = QtWidgets.QLabel(self.optionsPanel)
        self.dataReleaseText.setGeometry(QtCore.QRect(30, 20, 111, 31))
        self.dataReleaseText.setObjectName("dataReleaseText")
        self.dataReleaseComboBox = QtWidgets.QComboBox(self.optionsPanel)
        self.dataReleaseComboBox.setGeometry(QtCore.QRect(130, 20, 311, 31))
        self.dataReleaseComboBox.setObjectName("dataReleaseComboBox")
        self.dataReleaseComboBox.addItem("")
        self.dataReleaseComboBox.addItem("")
        
        self.anomBaseText = QtWidgets.QLabel(self.optionsPanel)
        self.anomBaseText.setGeometry(QtCore.QRect(30, 130, 191, 31))
        self.anomBaseText.setObjectName("anomBaseText") 
        self.anomBaseComboBox = QtWidgets.QComboBox(self.optionsPanel)
        self.anomBaseComboBox.setGeometry(QtCore.QRect(130, 130, 311, 31))
        self.anomBaseComboBox.setObjectName("anomBaseComboBox")
        self.anomBaseComboBox.addItem("")
        self.anomBaseComboBox.addItem("")
        self.anomDefText = QtWidgets.QLabel(self.optionsPanel)
        self.anomDefText.setGeometry(QtCore.QRect(30, 160, 121, 31))
        self.anomDefText.setObjectName("anomDefText")
        self.anomDefSpinBox = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.anomDefSpinBox.setGeometry(QtCore.QRect(160, 160, 68, 31))
        self.anomDefSpinBox.setDecimals(1)
        self.anomDefSpinBox.setMinimum(0.1)
        self.anomDefSpinBox.setMaximum(100.0)
        self.anomDefSpinBox.setSingleStep(0.1)
        self.anomDefSpinBox.setObjectName("anomDefSpinBox")
        self.anomDefComboBox = QtWidgets.QComboBox(self.optionsPanel)
        self.anomDefComboBox.setGeometry(QtCore.QRect(230, 160, 211, 31))
        self.anomDefComboBox.setObjectName("anomDefComboBox")
        self.anomDefComboBox.addItem("")
        self.anomDefComboBox.addItem("")
        
        self.datasetComboBox = QtWidgets.QComboBox(self.optionsPanel)
        self.datasetComboBox.setGeometry(QtCore.QRect(130, 50, 311, 31))
        self.datasetComboBox.setObjectName("datasetComboBox")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetComboBox.addItem("")
        self.datasetText = QtWidgets.QLabel(self.optionsPanel)
        self.datasetText.setGeometry(QtCore.QRect(30, 50, 111, 31))
        self.datasetText.setObjectName("datasetText")
        
        self.magnitudeText = QtWidgets.QLabel(self.optionsPanel)
        self.magnitudeText.setGeometry(QtCore.QRect(30, 80, 131, 31))
        self.magnitudeText.setObjectName("magnitudeText")
        self.magnitudeComboBox = QtWidgets.QComboBox(self.optionsPanel)
        self.magnitudeComboBox.setEnabled(True)
        self.magnitudeComboBox.setGeometry(QtCore.QRect(130, 80, 311, 31))
        self.magnitudeComboBox.setEditable(False)
        self.magnitudeComboBox.setMaxVisibleItems(11)
        self.magnitudeComboBox.setObjectName("magnitudeComboBox")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")
        self.magnitudeComboBox.addItem("")

        self.pressTracerText = QtWidgets.QLabel(self.optionsPanel)
        self.pressTracerText.setGeometry(QtCore.QRect(30, 200, 161, 31))
        self.pressTracerText.setObjectName("pressTracerText")
        self.pressTracerSpinBox = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.pressTracerSpinBox.setGeometry(QtCore.QRect(190, 200, 68, 31))
        self.pressTracerSpinBox.setDecimals(0)
        self.pressTracerSpinBox.setMinimum(1.0)
        self.pressTracerSpinBox.setMaximum(1000.0)
        self.pressTracerSpinBox.setSingleStep(1.0)
        self.pressTracerSpinBox.setProperty("value", 50.0)
        self.pressTracerSpinBox.setObjectName("pressTracerSpinBox")
        self.pressTracerUnitText = QtWidgets.QLabel(self.optionsPanel)
        self.pressTracerUnitText.setGeometry(QtCore.QRect(260, 200, 161, 31))
        self.pressTracerUnitText.setObjectName("pressTracerUnitText")
        
        self.latBandsText = QtWidgets.QLabel(self.optionsPanel)
        self.latBandsText.setGeometry(QtCore.QRect(480, 20, 171, 31))
        self.latBandsText.setObjectName("latBandsText")
        
        self.bandTropicsText = QtWidgets.QLabel(self.optionsPanel)
        self.bandTropicsText.setGeometry(QtCore.QRect(480, 50, 111, 31))
        self.bandTropicsText.setObjectName("bandTropicsText")
        self.bandTropicsPlusminus = QtWidgets.QLabel(self.optionsPanel)
        self.bandTropicsPlusminus.setGeometry(QtCore.QRect(550, 50, 21, 31))
        self.bandTropicsPlusminus.setObjectName("bandTropicsPlusminus")
        self.tropicsSpinBox = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.tropicsSpinBox.setGeometry(QtCore.QRect(560, 50, 68, 31))
        self.tropicsSpinBox.setDecimals(1)
        self.tropicsSpinBox.setMinimum(0.5)
        self.tropicsSpinBox.setMaximum(90.0)
        self.tropicsSpinBox.setSingleStep(0.5)
        self.tropicsSpinBox.setProperty("value", 23.5)
        self.tropicsSpinBox.setObjectName("tropicsSpinBox")
        
        self.band2Text = QtWidgets.QLabel(self.optionsPanel)
        self.band2Text.setGeometry(QtCore.QRect(480, 80, 111, 31))
        self.band2Text.setObjectName("band2Text")
        self.band2SpinBoxL = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band2SpinBoxL.setGeometry(QtCore.QRect(560, 80, 68, 31))
        self.band2SpinBoxL.setDecimals(1)
        self.band2SpinBoxL.setMinimum(0.0)
        self.band2SpinBoxL.setMaximum(90.0)
        self.band2SpinBoxL.setSingleStep(0.5)
        self.band2SpinBoxL.setProperty("value", 23.5)
        self.band2SpinBoxL.setObjectName("band2SpinBoxL")
        self.band2TextTo = QtWidgets.QLabel(self.optionsPanel)
        self.band2TextTo.setGeometry(QtCore.QRect(640, 80, 21, 31))
        self.band2TextTo.setObjectName("band2TextTo")
        self.band2SpinBoxR = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band2SpinBoxR.setGeometry(QtCore.QRect(670, 80, 68, 31))
        self.band2SpinBoxR.setDecimals(1)
        self.band2SpinBoxR.setMinimum(0.0)
        self.band2SpinBoxR.setMaximum(90.0)
        self.band2SpinBoxR.setSingleStep(0.5)
        self.band2SpinBoxR.setProperty("value", 35.0)
        self.band2SpinBoxR.setObjectName("band2SpinBoxR")
        
        self.band3Text = QtWidgets.QLabel(self.optionsPanel)
        self.band3Text.setGeometry(QtCore.QRect(480, 110, 111, 31))
        self.band3Text.setObjectName("band3Text")
        self.band3SpinBoxL = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band3SpinBoxL.setGeometry(QtCore.QRect(560, 110, 68, 31))
        self.band3SpinBoxL.setDecimals(1)
        self.band3SpinBoxL.setMinimum(0.0)
        self.band3SpinBoxL.setMaximum(90.0)
        self.band3SpinBoxL.setSingleStep(0.5)
        self.band3SpinBoxL.setProperty("value", 35.0)
        self.band3SpinBoxL.setObjectName("band3SpinBoxL")
        self.band3TextTo = QtWidgets.QLabel(self.optionsPanel)
        self.band3TextTo.setGeometry(QtCore.QRect(640, 110, 21, 31))
        self.band3TextTo.setObjectName("band3TextTo")
        self.band3SpinBoxR = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band3SpinBoxR.setGeometry(QtCore.QRect(670, 110, 68, 31))
        self.band3SpinBoxR.setDecimals(1)
        self.band3SpinBoxR.setMinimum(0.0)
        self.band3SpinBoxR.setMaximum(90.0)
        self.band3SpinBoxR.setSingleStep(0.5)
        self.band3SpinBoxR.setProperty("value", 66.5)
        self.band3SpinBoxR.setObjectName("band3SpinBoxR") 
        
        self.band4Text = QtWidgets.QLabel(self.optionsPanel)
        self.band4Text.setGeometry(QtCore.QRect(480, 140, 111, 31))
        self.band4Text.setObjectName("band4Text")
        self.band4SpinBoxL = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band4SpinBoxL.setGeometry(QtCore.QRect(560, 140, 68, 31))
        self.band4SpinBoxL.setDecimals(1)
        self.band4SpinBoxL.setMinimum(0.0)
        self.band4SpinBoxL.setMaximum(90.0)
        self.band4SpinBoxL.setSingleStep(0.5)
        self.band4SpinBoxL.setProperty("value", 66.5)
        self.band4SpinBoxL.setObjectName("band4SpinBoxL")
        self.band4TextTo = QtWidgets.QLabel(self.optionsPanel)
        self.band4TextTo.setGeometry(QtCore.QRect(640, 140, 21, 31))
        self.band4TextTo.setObjectName("band4TextTo")
        self.band4SpinBoxR = QtWidgets.QDoubleSpinBox(self.optionsPanel)
        self.band4SpinBoxR.setGeometry(QtCore.QRect(670, 140, 68, 31))
        self.band4SpinBoxR.setDecimals(1)
        self.band4SpinBoxR.setMinimum(0.0)
        self.band4SpinBoxR.setMaximum(90.0)
        self.band4SpinBoxR.setSingleStep(0.5)
        self.band4SpinBoxR.setProperty("value", 90.0)
        self.band4SpinBoxR.setObjectName("band4SpinBoxR")
        
        self.resetBandsButton = QtWidgets.QPushButton(self.optionsPanel)
        self.resetBandsButton.setEnabled(True)
        self.resetBandsButton.setGeometry(QtCore.QRect(560, 170, 161, 32))
        self.resetBandsButton.setObjectName("resetBandsButton")

        self.refreshTableButton = QtWidgets.QPushButton(self.optionsPanel)
        self.refreshTableButton.setEnabled(True)
        self.refreshTableButton.setGeometry(QtCore.QRect(20, 239, 221, 31))
        self.refreshTableButton.setStyleSheet("QPushButton{background-color: lightgreen; color: black;} "\
                                              "QPushButton::pressed{background-color : green;}")
        self.refreshTableButton.setObjectName("refreshTableButton")

        self.progressBar = QtWidgets.QProgressBar(self.optionsPanel)
        self.progressBar.setGeometry(QtCore.QRect(250, 250, 511, 20))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        
        # -------------------------------------------------------
        # -------------------- results panel --------------------
        
        self.resultsPanel = QtWidgets.QGroupBox(self.centralwidget)
        self.resultsPanel.setGeometry(QtCore.QRect(10, 280, 781, 361))
        self.resultsPanel.setObjectName("resultsPanel")
        self.resultsTable = QtWidgets.QTableWidget(self.resultsPanel)
        self.resultsTable.setEnabled(True)
        self.resultsTable.setGeometry(QtCore.QRect(10, 30, 761, 281))
        self.resultsTable.setAutoFillBackground(False)
        self.resultsTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.resultsTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.resultsTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.resultsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.resultsTable.setAutoScroll(True)
        self.resultsTable.setDragDropOverwriteMode(False)
        self.resultsTable.setAlternatingRowColors(True)
        self.resultsTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.resultsTable.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.resultsTable.setShowGrid(True)
        self.resultsTable.setGridStyle(QtCore.Qt.SolidLine)
        self.resultsTable.setCornerButtonEnabled(False)
        self.resultsTable.setRowCount(7)
        self.resultsTable.setColumnCount(6)
        self.resultsTable.setObjectName("resultsTable")
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.resultsTable.setHorizontalHeaderItem(5, item)
        self.resultsTable.horizontalHeader().setDefaultSectionSize(111)
        self.resultsTable.verticalHeader().setDefaultSectionSize(34)
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        self.resultsTable.verticalHeader().setStretchLastSection(True)
        
        self.exportCellsButton = QtWidgets.QPushButton(self.resultsPanel)
        self.exportCellsButton.setEnabled(True)
        self.exportCellsButton.setGeometry(QtCore.QRect(40, 320, 221, 32))
        self.exportCellsButton.setObjectName("exportCellsButton")
        self.exportAllCellsButton = QtWidgets.QPushButton(self.resultsPanel)
        self.exportAllCellsButton.setEnabled(True)
        self.exportAllCellsButton.setGeometry(QtCore.QRect(260, 320, 221, 32))
        self.exportAllCellsButton.setObjectName("exportAllCellsButton")

        self.exportProgressBar = QtWidgets.QProgressBar(self.resultsPanel)
        self.exportProgressBar.setGeometry(QtCore.QRect(480, 320, 131, 31))
        self.exportProgressBar.setProperty("value", 0)
        self.exportProgressBar.setInvertedAppearance(False)
        self.exportProgressBar.setObjectName("exportProgressBar")

        self.helpButton = QtWidgets.QPushButton(self.resultsPanel)
        self.helpButton.setEnabled(True)
        self.helpButton.setGeometry(QtCore.QRect(630, 320, 141, 31))
        self.helpButton.setStyleSheet("QPushButton{background-color: lightblue; color: black;} "\
                                      "QPushButton::pressed{background-color : blue;}")
        self.helpButton.setDefault(False)
        self.helpButton.setFlat(False)
        self.helpButton.setObjectName("helpButton")
        
        # -----------------------------------------------------
        # -------------------- plots panel --------------------
        
        self.plotPanel = QtWidgets.QGroupBox(self.centralwidget)
        self.plotPanel.setGeometry(QtCore.QRect(800, 0, 721, 641))
        self.plotPanel.setFlat(False)
        self.plotPanel.setObjectName("plotPanel")
        self.plotViewport = QtWidgets.QGraphicsView(self.plotPanel)
        self.plotViewport.setGeometry(QtCore.QRect(10, 30, 701, 601))
        self.plotViewport.setObjectName("plotViewport")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1541, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        # modify UI
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # set UI triggers
        self.setTriggers()


    # ==================================================================


    def retranslateUi(self, MainWindow):
        '''
        This function generated by QtDesigner; modifies elements of the GUI
        '''
        
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("HSW++ Zonal Statistics Tool")
        
        # -------------------------------------------------------
        # -------------------- options panel --------------------
        
        self.optionsPanel.setTitle(_translate("MainWindow", "Options"))
        
        self.dataReleaseText.setText(_translate("MainWindow", "Data Release:"))
        self.dataReleaseComboBox.setItemText(0, _translate("MainWindow", 
                                                "March release (low variability, 10 members)"))
        self.dataReleaseComboBox.setItemText(1, _translate("MainWindow", 
                                                "January release (high variability, 5 members)"))
        self.dataReleaseComboBox.setCurrentIndex(1)
        
        self.anomBaseText.setText(_translate("MainWindow", "Anomaly Base:"))
        self.anomBaseComboBox.setItemText(0, _translate("MainWindow", "Counterfactual"))
        self.anomBaseComboBox.setItemText(1, _translate("MainWindow", "Mean Climate"))
        self.anomBaseComboBox.setCurrentIndex(1)
        self.anomDefText.setText(_translate("MainWindow", "Anomaly Definition:"))
        self.anomDefComboBox.setItemText(0, _translate("MainWindow", "standard dev. from base"))
        self.anomDefComboBox.setItemText(1, _translate("MainWindow", "Kelvin from base"))
        self.anomDefComboBox.setCurrentIndex(0)
        
        self.datasetText.setText(_translate("MainWindow", "Dataset:"))
        self.datasetComboBox.setItemText(0, _translate("MainWindow", "ens_mean"))
        self.datasetComboBox.setItemText(1, _translate("MainWindow", "ens01"))
        self.datasetComboBox.setItemText(2, _translate("MainWindow", "ens02"))
        self.datasetComboBox.setItemText(3, _translate("MainWindow", "ens03"))
        self.datasetComboBox.setItemText(4, _translate("MainWindow", "ens04"))
        self.datasetComboBox.setItemText(5, _translate("MainWindow", "ens05"))
        self.datasetComboBox.setItemText(6, _translate("MainWindow", "ens06"))
        self.datasetComboBox.setItemText(7, _translate("MainWindow", "ens07"))
        self.datasetComboBox.setItemText(8, _translate("MainWindow", "ens08"))
        self.datasetComboBox.setItemText(9, _translate("MainWindow", "ens09"))
        self.datasetComboBox.setItemText(10, _translate("MainWindow", "ens10"))
        self.datasetComboBox.setCurrentIndex(1)
        
        self.magnitudeText.setText(_translate("MainWindow", "SO2 Magnitude:"))
        self.magnitudeComboBox.setItemText(0, _translate("MainWindow", "0.25X"))
        self.magnitudeComboBox.setItemText(1, _translate("MainWindow", "0.50X"))
        self.magnitudeComboBox.setItemText(2, _translate("MainWindow", "0.90X"))
        self.magnitudeComboBox.setItemText(3, _translate("MainWindow", "1.00X"))
        self.magnitudeComboBox.setItemText(4, _translate("MainWindow", "1.10X"))
        self.magnitudeComboBox.setItemText(5, _translate("MainWindow", "1.50X"))
        self.magnitudeComboBox.setItemText(6, _translate("MainWindow", "2.00X"))
        self.magnitudeComboBox.setCurrentIndex(3)

        self.pressTracerText.setText(_translate("MainWindow", "Pressure level for tracers:"))
        self.pressTracerUnitText.setText(_translate("MainWindow", "hPa"))
        
        self.latBandsText.setText(_translate("MainWindow", "Latitude Bands (degrees):"))
        
        self.bandTropicsText.setText(_translate("MainWindow", "Tropics:"))
        self.bandTropicsPlusminus.setText(_translate("MainWindow", "±"))
        
        self.band2Text.setText(_translate("MainWindow", "Band 2:"))
        self.band2TextTo.setText(_translate("MainWindow", "to"))
        
        self.band3Text.setText(_translate("MainWindow", "Band 3:"))
        self.band3TextTo.setText(_translate("MainWindow", "to"))
        
        self.band4Text.setText(_translate("MainWindow", "Band 4:"))
        self.band4TextTo.setText(_translate("MainWindow", "to"))
        
        self.resetBandsButton.setText(_translate("MainWindow", "reset to defaults"))

        self.refreshTableButton.setText(_translate("MainWindow", "refresh results table"))
        
        # -------------------------------------------------------
        # -------------------- results panel --------------------
                
        self.resultsPanel.setTitle(_translate("MainWindow", "Results"))
        item = self.resultsTable.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Band 4 NH"))
        item = self.resultsTable.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Band 3 NH"))
        item = self.resultsTable.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Band 2 NH"))
        item = self.resultsTable.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Band 1 Tropics"))
        item = self.resultsTable.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Band 2 SH"))
        item = self.resultsTable.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "Band 3 SH"))
        item = self.resultsTable.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "Band 4 SH"))
        item = self.resultsTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "SO2\n({} hPa)".format(self.pressTracerSpinBox.value())))
        item = self.resultsTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "SULFATE\n({} hPa)".format(self.pressTracerSpinBox.value())))
        item = self.resultsTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "AOD"))
        item = self.resultsTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "T025"))
        item = self.resultsTable.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "T050"))
        item = self.resultsTable.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "T1000"))
        
        self.exportCellsButton.setText(_translate("MainWindow", "export data for selected cell"))
        self.exportAllCellsButton.setText(_translate("MainWindow", "export data for all cells"))
        
        self.helpButton.setText(_translate("MainWindow", "HELP"))
        
        # -----------------------------------------------------
        # -------------------- plots panel --------------------
        
        self.plotPanel.setTitle(_translate("MainWindow", "Plots"))
    
    
    # ==================================================================
    
    
    def setTriggers(self):
        '''
        Sets up certain option choices to trigger other changes to the GUI
        '''
        
        # ---- allow data release selection to restrict other options
        self.dataReleaseComboBox.activated.connect(self.update_data_release_options)

        # ---- set up latitude band reset button
        defaultBands = [23.5, 23.5, 35.0, 35.0, 66.5, 66.5, 90.0]
        bandBoxes = [self.tropicsSpinBox, self.band2SpinBoxL, self.band2SpinBoxR, 
                     self.band3SpinBoxL, self.band3SpinBoxR, self.band4SpinBoxL, self.band4SpinBoxR]
        self.resetBandsButton.clicked.connect(lambda:  
            [bandBoxes[i].setValue(defaultBands[i]) for i in range(len(bandBoxes))])

        # ---- set up results refresh button
        self.refreshTableButton.clicked.connect(self.refresh_results)


    
    
    # ==================================================================

    
    def update_data_release_options(self, index):
        '''
        Updates options to only allow seletions available for the currently selected Data Release. This
        is needed since the January and March 2023 HSW++ data releases offer different numbers of ensemble
        members, and different SO2 mass magnitudes

        Parameters
        ----------
        index : int
            the index of the currently selected item in datReleaseComboBox
        '''
        ensMembersAvailable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        massMagsAvailable = [0.25, 0.50, 0.90, 1.00, 1.10, 1.50, 2.00]
        
        if index == 1: # January release (high variability, 5 members)
            enabledEnsMembers = [1, 2, 3, 4, 5]
            enabledMassMags = [1.00]
        elif index == 0: # March release (low variability, 10 members)
            enabledEnsMembers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            enabledMassMags = [0.25, 0.50, 0.90, 1.00, 1.10, 1.50, 2.00]

        for i in range(len(ensMembersAvailable)):
            if((i+1) in enabledEnsMembers):
                self.datasetComboBox.model().item(i+1).setEnabled(True)
            else:
                self.datasetComboBox.model().item(i+1).setEnabled(False)
        
        for i in range(len(massMagsAvailable)):
            if(massMagsAvailable[i] in enabledMassMags):
                self.magnitudeComboBox.model().item(i).setEnabled(True)
            else:
                self.magnitudeComboBox.model().item(i).setEnabled(False)
        
        return
    
    
    # ==================================================================

     
    def refresh_results(self):
        '''
        Refreshes the results table by initializing computation on the data, given
        the currently selected options
        '''
        
        _translate = QtCore.QCoreApplication.translate

        # ---- download data if needed
        self.progressBar.setProperty("value", 0)
        self.exportProgressBar.setProperty("value", 0)
        self.refreshTableButton.setText(_translate("MainWindow", "fetching data..."))
        QApplication.processEvents() 
        download_data(self.progressBar)
        
        # ---- gather currently selected options
        self.progressBar.setProperty("value", 0)
        self.exportProgressBar.setProperty("value", 0)
        self.refreshTableButton.setText(_translate("MainWindow", "working..."))
        QApplication.processEvents()
        
        data_release = ['030123', '011423'][self.dataReleaseComboBox.currentIndex()]
        dataset   = str(self.datasetComboBox.currentText())
        mass_mag  = str(self.magnitudeComboBox.currentText())
        anom_base = str(self.anomBaseComboBox.currentText())
        anom_def  = ['std', 'cf'][self.anomDefComboBox.currentIndex()]
        anom_n    = self.anomDefSpinBox.value()
        trac_pres = self.pressTracerSpinBox.value()

        band1_bounds = np.array([-self.tropicsSpinBox.value(), self.tropicsSpinBox.value()])
        band2_bounds = np.array([self.band2SpinBoxL.value(), self.band2SpinBoxR.value()])
        band3_bounds = np.array([self.band3SpinBoxL.value(), self.band3SpinBoxR.value()])
        band4_bounds = np.array([self.band4SpinBoxL.value(), self.band4SpinBoxR.value()])
        band_bounds  = np.array([band1_bounds, band2_bounds, band3_bounds, band4_bounds])

        # ---- call computation functions
        dh = data_handler(data_release, dataset, mass_mag, trac_pres, anom_base, anom_def, 
                          anom_n, band_bounds, self.progressBar, self.refreshTableButton)
        dh.load_data()
        dh.average_lat_bands(overwrite=True)
        dh.compute_anomalies()
        dh.compute_benchmark_values()
        dh.make_plots()

        # ---- done, set progress bar to 100 if not already done, reset button text
        self.progressBar.setProperty("value", 100)
        self.refreshTableButton.setText(_translate("MainWindow", "refresh results table"))
        
        # ---- update tracer table header text to match selected pressure level
        item = self.resultsTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "SO2\n({} hPa)".format(self.pressTracerSpinBox.value())))
        item = self.resultsTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "SULFATE\n({} hPa)".format(self.pressTracerSpinBox.value())))



# ======================================================================================
# ======================================================================================


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    print('\n\n ========== HSW++ Statistics Tool ==========\n\n')
    MainWindow.show()
    sys.exit(app.exec_())
