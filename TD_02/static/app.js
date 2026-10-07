// ==========================================================
// Global state
// ==========================================================

const configurationName =
    document.getElementById(
        "configuration-name"
    );

const selectedModelConfig =
    document.getElementById(
        "selected-model-config"
    );

const configEpochs =
    document.getElementById(
        "config-epochs"
    );

const configBatchSize =
    document.getElementById(
        "config-batch-size"
    );

const configLearningRate =
    document.getElementById(
        "config-learning-rate"
    );

const configMaxLength =
    document.getElementById(
        "config-max-length"
    );


let fineTunedModels = [];

let lossChart = null;
let trainingPoll = null;
let pipelineAnimation = null;

let fineTunedAvailable = false;


// ==========================================================
// DOM elements
// ==========================================================

const tabs = document.querySelectorAll(".tab");
const pages = document.querySelectorAll(".page");

const modelStatus = document.getElementById("model-status");

const trainButton = document.getElementById("train-button");
const trainingMessage = document.getElementById("training-message");

const trainingState = document.getElementById("training-state");
const currentEpoch = document.getElementById("current-epoch");
const currentLoss = document.getElementById("current-loss");
const progressBar = document.getElementById("progress-bar");

const translationInput = document.getElementById("translation-input");
const promptDisplay =
    document.getElementById("prompt-display");
const translationModel = document.getElementById("translation-model");
const fineTunedOption = document.getElementById("finetuned-option");
const translateButton = document.getElementById("translate-button");
const translationMessage = document.getElementById("translation-message");

const englishDisplay = document.getElementById("english-display");
const inputTokens = document.getElementById("input-tokens");
const outputTokens = document.getElementById("output-tokens");
const embeddingVectors =
    document.getElementById("embedding-vectors");

const embeddingDimension =
    document.getElementById("embedding-dimension");

const contextualVectors =
    document.getElementById("contextual-vectors");
const translationOutput = document.getElementById("translation-output");

const encoderStage = document.getElementById("encoder-stage");
const decoderStage = document.getElementById("decoder-stage");
const representationStage = document.getElementById(
    "representation-stage"
);

const pipelineSteps = [
    document.getElementById("pipeline-data"),
    document.getElementById("pipeline-tokenizer"),
    document.getElementById("pipeline-transformer"),
    document.getElementById("pipeline-loss"),
    document.getElementById("pipeline-update"),
];


// ==========================================================
// Utility
// ==========================================================

function sleep(milliseconds) {

    return new Promise(
        resolve => setTimeout(resolve, milliseconds)
    );
}


function setMessage(element, message, type = "") {

    element.textContent = message;

    element.classList.remove(
        "error",
        "success"
    );

    if (type) {
        element.classList.add(type);
    }
}


// ==========================================================
// Tabs
// ==========================================================

tabs.forEach(tab => {

    tab.addEventListener("click", () => {

        const targetPage = tab.dataset.page;

        tabs.forEach(item => {
            item.classList.remove("active");
        });

        pages.forEach(page => {
            page.classList.remove("active");
        });

        tab.classList.add("active");

        document
            .getElementById(targetPage)
            .classList.add("active");
    });

});


// ==========================================================
// Loss chart
// ==========================================================

function initializeLossChart() {

    const context = document
        .getElementById("loss-chart")
        .getContext("2d");


    lossChart = new Chart(
        context,
        {
            type: "line",

            data: {
                labels: [],

                datasets: [
                    {
                        label: "Training loss",
                        data: [],
                        borderWidth: 2,
                        pointRadius: 2,
                        tension: 0.15,
                    }
                ]
            },

            options: {

                responsive: true,
                maintainAspectRatio: false,

                animation: false,

                interaction: {
                    intersect: false,
                    mode: "index",
                },

                plugins: {

                    legend: {
                        display: false,
                    }

                },

                scales: {

                    x: {
                        title: {
                            display: true,
                            text: "Epoch",
                        }
                    },

                    y: {
                        title: {
                            display: true,
                            text: "Loss",
                        },

                        beginAtZero: true,
                    }

                }

            }
        }
    );
}


// ==========================================================
// Model information
// ==========================================================

async function loadModelInfo() {

    try {

        const response = await fetch(
            "/api/model-info"
        );

        const data = await response.json();


        if (!response.ok) {
            throw new Error(
                data.error || "Could not load model information."
            );
        }


        // fineTunedAvailable =
        //     data.fine_tuned_available;

        fineTunedModels =
            data.finetuned_models;

        modelStatus.textContent =
            `${data.name} · ` +
            `${data.parameters.toLocaleString()} parameters · ` +
            `${data.dataset_size} sentence pairs`;


        // updateFineTunedOption();
        populateModelDropdown();


    } catch (error) {

        modelStatus.textContent =
            "Could not load model information.";

        console.error(error);
    }
}

function populateModelDropdown() {

    translationModel.innerHTML = "";


    const pretrainedOption =
        document.createElement("option");

    pretrainedOption.value = "";

    pretrainedOption.textContent =
        "Original pretrained T5";

    translationModel.appendChild(
        pretrainedOption
    );


    fineTunedModels.forEach(
        config => {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                config.name;

            option.textContent =
                `Fine-tuned — ${config.name}`;

            translationModel.appendChild(
                option
            );
        }
    );


    updateSelectedModelConfig();
}

function updateSelectedModelConfig() {

    const name =
        translationModel.value;


    if (!name) {

        selectedModelConfig
            .classList
            .add("hidden");

        return;
    }


    const config =
        fineTunedModels.find(
            model => model.name === name
        );


    if (!config) {
        return;
    }


    configEpochs.textContent =
        config.epochs;

    configBatchSize.textContent =
        config.batch_size;

    configLearningRate.textContent =
        config.learning_rate;

    configMaxLength.textContent =
        config.max_length;


    selectedModelConfig
        .classList
        .remove("hidden");
}


translationModel.addEventListener(
    "change",
    updateSelectedModelConfig
);

// ==========================================================
// Training pipeline animation
// ==========================================================

function startPipelineAnimation() {

    stopPipelineAnimation();

    let currentStep = 0;


    function activateStep() {

        pipelineSteps.forEach(step => {
            step.classList.remove(
                "active-step"
            );
        });


        pipelineSteps[
            currentStep
        ].classList.add(
            "active-step"
        );


        currentStep =
            (currentStep + 1)
            % pipelineSteps.length;
    }


    activateStep();

    pipelineAnimation = setInterval(
        activateStep,
        550
    );
}


function stopPipelineAnimation() {

    if (pipelineAnimation !== null) {

        clearInterval(
            pipelineAnimation
        );

        pipelineAnimation = null;
    }


    pipelineSteps.forEach(step => {
        step.classList.remove(
            "active-step"
        );
    });
}


// ==========================================================
// Start training
// ==========================================================

trainButton.addEventListener(
    "click",
    async () => {

        const name =
            configurationName.value.trim();

        const epochs = Number(
            document.getElementById(
                "epochs"
            ).value
        );

        const batchSize = Number(
            document.getElementById(
                "batch-size"
            ).value
        );

        const learningRate = Number(
            document.getElementById(
                "learning-rate"
            ).value
        );

        const maxLength = Number(
            document.getElementById(
                "max-length"
            ).value
        );


        setMessage(
            trainingMessage,
            ""
        );

        if (!name) {

            setMessage(
                trainingMessage,
                "Choose a configuration name before training.",
                "error"
            );

            configurationName.focus();

            return;
        }

        try {

            const response = await fetch(
                "/api/train",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        name: name,
                        epochs: epochs,
                        batch_size: batchSize,
                        learning_rate: learningRate,
                        max_length: maxLength,
                    }),
                }
            );


            const data = await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Could not start training."
                );
            }


            // Reset chart.

            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.update();


            currentEpoch.textContent =
                `0 / ${epochs}`;

            currentLoss.textContent =
                "—";

            progressBar.style.width =
                "0%";


            trainingState.textContent =
                "Training";

            trainingState.className =
                "state-badge training";


            trainButton.disabled = true;


            setMessage(
                trainingMessage,
                "Fine-tuning has started."
            );


            startPipelineAnimation();
            startTrainingPolling();


        } catch (error) {

            setMessage(
                trainingMessage,
                error.message,
                "error"
            );

        }

    }
);


// ==========================================================
// Poll training status
// ==========================================================

function startTrainingPolling() {

    if (trainingPoll !== null) {

        clearInterval(
            trainingPoll
        );
    }


    updateTrainingStatus();


    trainingPoll = setInterval(
        updateTrainingStatus,
        700
    );
}


async function updateTrainingStatus() {

    try {

        const response = await fetch(
            "/api/training-status"
        );

        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                "Could not retrieve training status."
            );
        }


        // ----------------------------------------------
        // Epoch and loss
        // ----------------------------------------------

        currentEpoch.textContent =
            `${data.epoch} / ${data.total_epochs}`;


        if (data.loss !== null) {

            currentLoss.textContent =
                Number(data.loss).toFixed(4);

        } else {

            currentLoss.textContent =
                "—";
        }


        // ----------------------------------------------
        // Progress bar
        // ----------------------------------------------

        let progress = 0;

        if (data.total_epochs > 0) {

            progress =
                (
                    data.epoch
                    / data.total_epochs
                )
                * 100;
        }


        progressBar.style.width =
            `${progress}%`;


        // ----------------------------------------------
        // Loss graph
        // ----------------------------------------------

        lossChart.data.labels =
            data.history.map(
                point => point.epoch
            );


        lossChart.data.datasets[0].data =
            data.history.map(
                point => point.loss
            );


        lossChart.update();


        // ----------------------------------------------
        // Error
        // ----------------------------------------------

        if (data.error) {

            stopTrainingPolling();
            stopPipelineAnimation();

            trainButton.disabled = false;

            trainingState.textContent =
                "Error";

            trainingState.className =
                "state-badge error";


            setMessage(
                trainingMessage,
                data.error,
                "error"
            );

            return;
        }


        // ----------------------------------------------
        // Finished
        // ----------------------------------------------

        if (data.finished) {

            stopTrainingPolling();
            stopPipelineAnimation();

            trainButton.disabled = false;

            trainingState.textContent =
                "Finished";

            trainingState.className =
                "state-badge finished";


            progressBar.style.width =
                "100%";


            setMessage(
                trainingMessage,
                "Fine-tuning complete. The model has been saved.",
                "success"
            );


            await loadModelInfo();

            return;
        }


        // ----------------------------------------------
        // Still training
        // ----------------------------------------------

        if (data.running) {

            trainingState.textContent =
                "Training";

            trainingState.className =
                "state-badge training";
        }


    } catch (error) {

        console.error(error);

    }
}


function stopTrainingPolling() {

    if (trainingPoll !== null) {

        clearInterval(
            trainingPoll
        );

        trainingPoll = null;
    }
}


// ==========================================================
// Reset translation visualization
// ==========================================================

function resetTranslationVisualization() {

    englishDisplay.textContent = "—";

    inputTokens.innerHTML = "";
    promptDisplay.textContent = "—";
    outputTokens.innerHTML = "";

    embeddingVectors.innerHTML = "";
    contextualVectors.innerHTML = "";

    embeddingDimension.textContent = "";

    translationOutput.textContent = "—";


    encoderStage.classList.remove(
        "processing"
    );

    decoderStage.classList.remove(
        "processing"
    );
}


// ==========================================================
// Create token
// ==========================================================

function createToken(
    token,
    tokenId,
    generated = false,
    delay = 0
) {

    const element =
        document.createElement("span");

    element.classList.add("token");

    if (generated) {
        element.classList.add("generated");
    }


    const tokenText =
        document.createElement("span");

    tokenText.classList.add("token-text");
    tokenText.textContent = token;


    const tokenIdElement =
        document.createElement("span");

    tokenIdElement.classList.add("token-id");
    tokenIdElement.textContent = tokenId;


    element.appendChild(tokenText);
    element.appendChild(tokenIdElement);

    element.style.animationDelay =
        `${delay}ms`;

    return element;
}


function createVectorCard(
    token,
    vector,
    contextual = false,
    delay = 0
) {

    const card =
        document.createElement("div");

    card.classList.add(
        "vector-card"
    );


    if (contextual) {

        card.classList.add(
            "contextual-vector"
        );
    }


    card.style.animationDelay =
        `${delay}ms`;


    // Token name

    const tokenLabel =
        document.createElement("div");

    tokenLabel.classList.add(
        "vector-token"
    );

    tokenLabel.textContent =
        token;


    // Vector

    const values =
        document.createElement("div");

    values.classList.add(
        "vector-values"
    );


    const first =
        document.createElement("span");

    first.textContent =
        Number(vector.first).toFixed(3);


    const second =
        document.createElement("span");

    second.textContent =
        Number(vector.second).toFixed(3);


    const ellipsis =
        document.createElement("span");

    ellipsis.classList.add(
        "vector-ellipsis"
    );

    ellipsis.textContent =
        "⋮";


    const last =
        document.createElement("span");

    last.textContent =
        Number(vector.last).toFixed(3);


    values.appendChild(first);
    values.appendChild(second);
    values.appendChild(ellipsis);
    values.appendChild(last);


    card.appendChild(
        tokenLabel
    );

    card.appendChild(
        values
    );


    return card;
}

// ==========================================================
// Translation
// ==========================================================

translateButton.addEventListener(
    "click",
    async () => {

        const sentence =
            translationInput.value.trim();


        if (!sentence) {

            setMessage(
                translationMessage,
                "Enter an English sentence first.",
                "error"
            );

            return;
        }


        const modelName =
            translationModel.value || null;


        resetTranslationVisualization();


        setMessage(
            translationMessage,
            ""
        );


        translateButton.disabled =
            true;


        try {

            // ------------------------------------------
            // English sentence
            // ------------------------------------------

            englishDisplay.textContent =
                sentence;

            await sleep(500);


            // ------------------------------------------
            // Tokenization
            // ------------------------------------------

            const tokenResponse =
                await fetch(
                    "/api/tokenize",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",
                        },

                        body: JSON.stringify({
                            sentence: sentence,
                            model_name: modelName,
                        }),
                    }
                );


            const tokenData =
                await tokenResponse.json();

            promptDisplay.textContent =
                tokenData.prompt;

            await sleep(600);


            if (!tokenResponse.ok) {

                throw new Error(
                    tokenData.error ||
                    "Tokenization failed."
                );
            }


            inputTokens.innerHTML =
                "";


            tokenData.tokens.forEach(
                (token, index) => {

                    const element =
                        createToken(
                            token,
                            tokenData.token_ids[index],
                            false,
                            index * 70
                        );

                    inputTokens.appendChild(
                        element
                    );
                }
            );


            await sleep(
                500
                + tokenData.tokens.length * 70
            );


            // ------------------------------------------
            // Input embeddings
            // ------------------------------------------

            embeddingVectors.innerHTML = "";

            embeddingDimension.textContent =
                `Each token is represented by a ` +
                `${tokenData.embedding_dimension}-dimensional vector`;


            tokenData.embeddings.forEach(
                (vector, index) => {

                    const card =
                        createVectorCard(
                            tokenData.tokens[index],
                            vector,
                            false,
                            index * 80
                        );

                    embeddingVectors.appendChild(
                        card
                    );
                }
            );


            await sleep(
                600
                + tokenData.embeddings.length * 80
            );

            // ------------------------------------------
            // Encoder
            // ------------------------------------------

            encoderStage.classList.add(
                "processing"
            );

            await sleep(1300);

            encoderStage.classList.remove(
                "processing"
            );

            // ------------------------------------------
            // Contextual representations
            // ------------------------------------------

            contextualVectors.innerHTML = "";


            tokenData.contextual_vectors.forEach(
                (vector, index) => {

                    const card =
                        createVectorCard(
                            tokenData.tokens[index],
                            vector,
                            true,
                            index * 80
                        );

                    contextualVectors.appendChild(
                        card
                    );
                }
            );


            await sleep(
                600
                + tokenData.contextual_vectors.length * 80
            );


            // ------------------------------------------
            // Contextual representations
            // ------------------------------------------

            // representationStage.classList.add(
            //     "active"
            // );

            await sleep(1000);


            // ------------------------------------------
            // Translation request
            // ------------------------------------------

            decoderStage.classList.add(
                "processing"
            );


            const translationResponse =
                await fetch(
                    "/api/translate",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",
                        },

                        body: JSON.stringify({
                            sentence: sentence,
                            model_name: modelName,
                            max_new_tokens: 40,
                        }),
                    }
                );


            const translationData =
                await translationResponse.json();


            if (!translationResponse.ok) {

                throw new Error(
                    translationData.error ||
                    "Translation failed."
                );
            }


            // ------------------------------------------
            // Show generated tokens sequentially
            // ------------------------------------------

            outputTokens.innerHTML =
                "";


            for (
                let index = 0;
                index < translationData.tokens.length;
                index++
            ) {

                const token =
                    translationData.tokens[index];


                const element =
                    createToken(
                        token,
                        translationData.token_ids[index],
                        true
                    );


                outputTokens.appendChild(
                    element
                );


                await sleep(300);
            }


            decoderStage.classList.remove(
                "processing"
            );

            // representationStage.classList.remove(
            //     "active"
            // );


            // ------------------------------------------
            // Final decoded sentence
            // ------------------------------------------

            await sleep(300);


            translationOutput.textContent =
                translationData.translation;


            setMessage(
                translationMessage,
                modelName
                    ? `Translation generated with fine-tuned model "${modelName}".`
                    : "Translation generated with the original pretrained model.",
                "success"
            );


        } catch (error) {

            decoderStage.classList.remove(
                "processing"
            );

            encoderStage.classList.remove(
                "processing"
            );

            // representationStage.classList.remove(
            //     "active"
            // );


            setMessage(
                translationMessage,
                error.message,
                "error"
            );

        } finally {

            translateButton.disabled =
                false;

        }

    }
);


// ==========================================================
// Allow Enter to translate
// ==========================================================

translationInput.addEventListener(
    "keydown",
    event => {

        if (event.key === "Enter") {

            translateButton.click();

        }

    }
);


// ==========================================================
// Initialize application
// ==========================================================

initializeLossChart();
loadModelInfo();
resetTranslationVisualization();