"""
Algoritmo #15: Transformer (Attention Mechanism)
State-of-the-art: arquitectura basada en atención para secuencias
"""

import numpy as np
import os

# Importar TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no instalado.")


class TransformerBlock(layers.Layer):
    """
    Bloque Transformer: Multi-Head Attention + Feed Forward
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerLottery:
    """
    Transformer para predicción de lotería.
    
    Arquitectura:
    - Input: Secuencia de últimos N sorteos
    - Positional Encoding: información de orden temporal
    - Transformer Blocks: 2 capas con 4-head attention
    - Output: 56 probabilidades (multi-label)
    
    Ventaja sobre LSTM:
    - Atención permite capturar dependencias de largo alcance
    - Procesamiento paralelo (más rápido que LSTM)
    
    Hipótesis:
    - Si hay patrones complejos en secuencias, Transformer los detectará
    - Esperamos: mismo overfitting que LSTM, performance = azar
    """
    
    def __init__(self, sequence_length=20, embed_dim=32, num_heads=4, ff_dim=64, epochs=30):
        """
        Args:
            sequence_length: Longitud de secuencia de entrada
            embed_dim: Dimensión de embeddings
            num_heads: Número de attention heads
            ff_dim: Dimensión de feed-forward layer
            epochs: Épocas de entrenamiento
        """
        self.name = f"Transformer (heads={num_heads}, seq={sequence_length})"
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.epochs = epochs
        self.model = None
        self.history = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no disponible.")
    
    def _create_sequences(self, history):
        """
        Crea secuencias de entrenamiento
        """
        X = []
        y = []
        
        all_draws = [draw['numbers'] for draw in history]
        
        for i in range(self.sequence_length, len(all_draws)):
            sequence = all_draws[i - self.sequence_length:i]
            X.append(sequence)
            
            # Target: multi-hot encoding
            target = np.zeros(56)
            for num in all_draws[i]:
                target[num - 1] = 1
            y.append(target)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def _build_model(self):
        """
        Construye arquitectura Transformer
        """
        inputs = layers.Input(shape=(self.sequence_length, 6))
        
        # Proyección a espacio embedding
        x = layers.Dense(self.embed_dim)(inputs)
        
        # Positional encoding (simple: learned embeddings)
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        pos_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.embed_dim
        )(positions)
        
        x = x + pos_embedding
        
        # Transformer blocks
        x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(x)
        x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(x)
        
        # Global average pooling (reduce sequence)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: 56 probabilidades
        outputs = layers.Dense(56, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def fit(self, history):
        """
        Entrena Transformer
        """
        if len(history) < self.sequence_length + 5:
            raise ValueError(f"❌ Historial insuficiente. Necesitas {self.sequence_length + 5}+ sorteos.")
        
        X, y = self._create_sequences(history)
        
        print(f"✅ {self.name}: Preparando entrenamiento...")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Sequence shape: {X.shape[1:]}")
        
        self.model = self._build_model()
        
        print(f"\n🏗️  Arquitectura:")
        self.model.summary(print_fn=lambda x: print(f"   {x}"))
        
        print(f"\n🎓 Entrenando {self.epochs} épocas...")
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=4,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"✅ Entrenamiento completado")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Val Loss: {final_val_loss:.4f}")
        
        if final_val_loss > final_loss * 1.5:
            print(f"   ⚠️ Overfitting detectado")
        
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo
        """
        if self.model is None:
            raise ValueError("❌ Modelo no entrenado.")
        
        all_draws = [draw['numbers'] for draw in history]
        sequence = all_draws[-self.sequence_length:]
        
        X_pred = np.array([sequence], dtype=np.float32)
        
        probabilities = self.model.predict(X_pred, verbose=0)[0]
        
        top_indices = np.argsort(probabilities)[-6:]
        prediction = sorted(top_indices + 1)
        
        return prediction


# ==================== TEST ====================
if __name__ == "__main__":
    print("🤖 ALGORITMO TRANSFORMER - TEST")
    print("=" * 60)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow no disponible.")
        exit(1)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    from src.data.collector import MelateCollector
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    if len(history) < 25:
        print(f"⚠️ Datos insuficientes ({len(history)} sorteos).")
        print("   Transformer requiere 25+ sorteos.")
        print("   Continuando de todas formas...")
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Entrenar
    print("\n" + "="*60)
    print("🔮 TEST 1: Entrenamiento Transformer")
    print("="*60)
    
    transformer_model = TransformerLottery(
        sequence_length=10,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        epochs=20
    )
    transformer_model.fit(history)
    
    # Test 2: Predicción
    print("\n" + "="*60)
    print("🔮 TEST 2: Predicción")
    print("="*60)
    
    prediction = transformer_model.predict(history)
    print(f"\n🎯 Predicción: {prediction}")
    
    # Test 3: Validación (1 sorteo por velocidad)
    print("\n" + "="*60)
    print("📊 VALIDACIÓN (último sorteo)")
    print("="*60)
    
    if len(history) >= 25:
        train_data = history[:-1]
        test_result = history[-1]['numbers']
        
        transformer_val = TransformerLottery(
            sequence_length=10,
            epochs=15
        )
        transformer_val.fit(train_data)
        pred_val = transformer_val.predict(train_data)
        
        matches = len(set(pred_val) & set(test_result))
        
        print(f"\nSorteo {history[-1]['date']}:")
        print(f"   Predicción:  {pred_val}")
        print(f"   Real:        {test_result}")
        print(f"   ✅ Aciertos: {matches}/6")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - Transformer usa attention para capturar dependencias")
    print("   - Más moderno que LSTM (state-of-the-art en NLP)")
    print("   - Pero lotería no tiene dependencias reales")
    print("   - Performance esperada: ~1.0 aciertos (igual que LSTM)")
    print("   - Conclusión: Más complejidad ≠ mejor predicción en datos aleatorios")