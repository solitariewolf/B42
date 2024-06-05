function enviarMensagem() {
    var mensagem = document.getElementById('mensagem').value.trim();

    if (mensagem) {
        document.getElementById('mensagem').value = '';
        document.getElementById('enviar').disabled = true; // Desabilitar o botão

        fetch('http://localhost:8000/responder', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ mensagem: mensagem })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Problema na resposta do servidor');
            }
            return response.json();
        })
        .then(data => {
            var display = document.getElementById('message-display');
            var perguntaUsuario = document.createElement('p');
            perguntaUsuario.textContent = 'Você: ' + mensagem;
            display.appendChild(perguntaUsuario);

            var respostaBot = document.createElement('p');
            respostaBot.textContent = 'Bot: ' + data.resposta;
            display.appendChild(respostaBot);

            display.scrollTop = display.scrollHeight;
        })
        .catch(error => {
            console.error('Erro ao enviar mensagem:', error);
            alert('Desculpe, houve um erro ao enviar sua mensagem.');
        })
        .finally(() => {
            document.getElementById('enviar').disabled = false; // Reabilitar o botão
        });
    } else {
        alert('Por favor, digite uma mensagem.');
    }
}

document.getElementById('enviar').addEventListener('click', enviarMensagem);

document.getElementById('mensagem').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        enviarMensagem();
    }
});
