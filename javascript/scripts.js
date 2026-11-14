function getCurrentExtraSourceImg_moebius(dummy_component, imgCom) {
    const img = gradioApp().querySelector('#extras_image div div img');
    const maskEl = gradioApp().getElementById('moebius_extras_mask');
    const removeButton = maskEl ? maskEl.querySelector('button[aria-label="Remove Image"]') : null;
    if (removeButton) {
        removeButton.click();
    }
    return img ? [img.src] : [null];
}

function moebiusApplyZoomAndPanIntegration() {
    if (typeof window.applyZoomAndPanIntegration === "function") {
        window.applyZoomAndPanIntegration("#moebius_extras", ["#moebius_extras_mask"]);
        var index = uiUpdateCallbacks.indexOf(moebiusApplyZoomAndPanIntegration);
        if (index !== -1) {
            uiUpdateCallbacks.splice(index, 1);
        }
    }
}

onUiUpdate(moebiusApplyZoomAndPanIntegration);
