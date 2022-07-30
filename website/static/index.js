function deletePredict(predictId) {
    fetch("/delete-predict", {
      method: "POST",
      body: JSON.stringify({ predictId: predictId }),
    }).then((_res) => {
      window.location.href = "/histori";
    });
  }