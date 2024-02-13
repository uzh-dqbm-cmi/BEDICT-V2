// Variables to store the selected editor, prediction type, and screening methods
var selectedEditorInVitro = "";
var selectedEditorInVivo = "";
var selectedcelltype = "";
var selectedPredictionType = ""; // Variable to store the selected prediction type

function selectPredictionType(predictionType) {
    var inVitroSection = document.getElementById("inVitroSection");
    var inVivoSection = document.getElementById("inVivoSection");
    var cellLineContainer = document.getElementById("cell_line");

    // Update the selected prediction type and change button color
    selectedPredictionType = predictionType;
    updateSelectedButtonStyle(selectedPredictionType, "predictionTypeButtons");

    // Hide/show sections based on the selected prediction type
    if (predictionType === 'InVitro') {
        inVitroSection.classList.remove("hidden");
        inVivoSection.classList.add("hidden");
    } else if (predictionType === 'InVivo') {
        inVivoSection.classList.remove("hidden");
        inVitroSection.classList.add("hidden");
        // Get the selected cell line
        var selectedCellLine = getSelectedCellLine();
        generateEditorButtonsForCellLine(selectedCellLine);
    }
}

function selectCellLine(cellLine) {
    // Set the selected cell line
    selectedcelltype = cellLine;

    // Update the selected button style
    updateSelectedButtonStyle(selectedcelltype, "cell_line_buttons");

    // Generate editor buttons for the selected cell line
    generateEditorButtonsForCellLine(selectedcelltype);

    // Show the editor buttons container
    var inVivoEditorButtonsContainer = document.getElementById("inVivoEditorButtonsContainer");
    inVivoEditorButtonsContainer.classList.remove("hidden");
}

function getSelectedCellLine() {
    var cellLineContainer = document.getElementById("cell_line");
    if (cellLineContainer != null) {
        var selectedCellLine = "";
        cellLineContainer.childNodes.forEach(function (label) {
            if (label.tagName === "LABEL" && label.classList.contains("selected")) {
                selectedCellLine = label.textContent.trim();
            }
        });
    }
    return selectedCellLine;
}

function selectEditor(editorName, predictionType) {
    console.log(`Selected Editor: ${editorName}, Prediction Type: ${predictionType}`);
    if (predictionType === 'InVitro') {
        selectedEditorInVitro = editorName;
        updateSelectedButtonStyle(selectedEditorInVitro, "inVitroEditorButtons");
    } else if (predictionType === 'InVivo') {
        selectedEditorInVivo = editorName;
        // inVivoEditorButtons
        updateSelectedButtonStyle(selectedEditorInVivo, "inVivoEditorButtons");
    }
}


function generateEditorButtonsForCellLine(cellLine) {
    var editorButtonsContainer = document.getElementById("inVivoEditorButtonsContainer")
    editorButtonsContainer.innerHTML = ""; // Clear previous buttons

    // Example editors for each cell line (customize as needed)
    var editors = {
        'Liver_LentiAAV': ['ABE8e-SpRY', 'ABEmax-SpRY'],
        'Liver_LentiLNP': ['ABE8e-SpRY', 'ABE8e-NG', 'ABE8e-SpCas9', 'ABEmax-SpRY'],
        'Liver_SBApproach': ['ABE8e-SpRY', 'ABEmax-SpRY']
        // Add more cell lines and editors as needed
    };

    // Create buttons for each editor associated with the selected cell line
    if (cellLine in editors) {
        editors[cellLine].forEach(function (editor) {
            var button = document.createElement("button");
            button.type = "button";
            button.textContent = editor;
            button.onclick = function () {
                selectEditor(editor, 'InVivo');
            };
            editorButtonsContainer.appendChild(button);
        });
    }
    // Show the editor buttons container
    document.getElementById("inVivoEditorButtonsContainer").classList.remove("hidden");
}


function handleScreeningMethodChange(checkboxId) {
    var checkbox = document.getElementById(checkboxId);

    if (checkbox.checked) {
        selectedScreeningMethods.push(checkboxId);
    } else {
        var index = selectedScreeningMethods.indexOf(checkboxId);
        if (index !== -1) {
            selectedScreeningMethods.splice(index, 1);
        }
    }

    // Show/hide editor selection based on the selected screening methods
    var inVivoEditorButtons = document.getElementById("inVivoEditorButtons");
    if (selectedScreeningMethods.length > 0) {
        inVivoEditorButtons.classList.remove("hidden");
    } else {
        inVivoEditorButtons.classList.add("hidden");
    }
}

function updateSelectedButtonStyle(selectedItem, containerId) {
    var buttons = document.querySelectorAll(`#${containerId} button`);
    buttons.forEach(function (button) {
        button.classList.remove("selected");
    });

    for (var i = 0; i < buttons.length; i++) {
        if (buttons[i].textContent.replaceAll(' ','') === selectedItem) {
            buttons[i].classList.add("selected");
            break;
        }
    }
}

function predict() {
    var inVitroSection = document.getElementById("inVitroSection");
    var inVivoSection = document.getElementById("inVivoSection");

    // Determine the prediction type
    var predictionType = inVitroSection.classList.contains("hidden") ? "InVivo" : "InVitro";

    // Based on the prediction type, get the appropriate sequence, selected editor, and screening methods
    var sequence, selectedEditor, celltype;

    if (predictionType === 'InVitro') {
        sequence = document.getElementById("inVitroSequenceInput").value;
        selectedEditor = selectedEditorInVitro;
    } else if (predictionType === 'InVivo') {
        sequence = document.getElementById("inVivoSequenceInput").value;
        selectedEditor = selectedEditorInVivo;
        celltype = selectedcelltype
    }

    // Check if an editor is selected and sequence is provided
    if (!selectedEditor) {
        alert("Please select an editor and enter a DNA sequence of length 24.");
        return;
    }

    // Make an AJAX request to Flask server for prediction
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    xhr.onload = function () {
        if (xhr.status === 200) {
            var predResultElement = document.getElementById("predictionResults");
            var responseJSON = JSON.parse(xhr.responseText);
            htmlJSON = responseJSON['html'];
            var keys = Object.keys(htmlJSON);
            var htmlToPrint= "";
            for (var cKey in keys) {
                htmlToPrint += htmlJSON[keys[cKey]]+"<br>";
            }



            // var htmlResponse = xhr.response.replaceAll('\\n','\n');
            // htmlResponse = htmlResponse.replaceAll('\\"','"');

            predResultElement.innerHTML = htmlToPrint;

    //         var results = JSON.parse(xhr.responseText);
                prepareAndRenderCVSDownload(responseJSON['predictionlist']);
    //         // saveTemplateAsFile("filename.json", myDataObj);
            // $("#predictionResults").html(xhr.result)   
            // alert("received results")
        } else {
            alert("Error in prediction.");
        }
    };

    xhr.send(JSON.stringify({ "input_data": sequence, "editor_name": selectedEditor, "prediction_type": predictionType, "cell_type": celltype}));
}

const saveTemplateAsFile = (filename, dataObjToWrite) => {
    const blob = new Blob([JSON.stringify(dataObjToWrite)], { type: "text/json" });
    const link = document.createElement("a");

    link.download = filename;
    link.href = window.URL.createObjectURL(blob);
    link.dataset.downloadurl = ["text/json", link.download, link.href].join(":");

    const evt = new MouseEvent("click", {
        view: window,
        bubbles: true,
        cancelable: true,
    });

    link.dispatchEvent(evt);
    link.remove()
};



/** Download contents as a file
 * Source: https://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side
 */
function downloadBlob(content, filename, contentType) {
    // Create a blob
    var blob = new Blob([content], { type: contentType });
    var url = URL.createObjectURL(blob);

    // Create a link to download it
    var pom = document.createElement('a');
    pom.href = url;
    pom.setAttribute('download', filename);
    pom.click();
}

/** Convert a 2D array into a CSV string
*/
function arrayToCsv(data) {
    return data.map(row =>
        row
            .map(String)  // convert every value to String
            .map(v => v.replaceAll('"', '""'))  // escape double quotes
            .map(v => `"${v}"`)  // quote it
            .join(',')  // comma-separated
    ).join('\r\n');  // rows starting on new lines
}

function prepareAndRenderCVSDownload(results) {
    var resultsDiv = document.getElementById("predictionResults");
    // resultsDiv.innerHTML = "";

    // var table = document.createElement("table");
    // var headerRow = table.insertRow();
    var headers = ["Index", "Target Sequence", "Output Sequence", "Probability"];

    var csvArray = Array();
    csvArray.push(headers);

    // headers.forEach(function (header) {
        // var cell_line = headerRow.insertCell();
        // cell_line.textContent = header;
    // });
    results.forEach(function (entry) {
        // var row = table.insertRow();
        var cell_lines = [entry.index, entry.Ref_seq, entry.Output_seq, entry.score.toFixed(3)];
        csvArray.push(cell_lines);

        // cell_lines.forEach(function (cell_lineValue) {
            // var cell_line = row.insertCell();
            // cell_line.textContent = cell_lineValue;
        // });
    });

    var downloadButton = document.createElement("button");
    downloadButton.data = "Download Result";
    downloadButton.innerHTML = 'Download Result as csv';

    downloadButton.onclick = function () {
        // saveTemplateAsFile("result.json", results);
        csv = arrayToCsv(csvArray);
        downloadBlob(csv, 'export.csv', 'text/csv;charset=utf-8;')
    };
    // resultsDiv.appendChild(table);
    resultsDiv.appendChild(downloadButton);

}
