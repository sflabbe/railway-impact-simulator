# Audit Report: Newton Solver Non-Convergence Investigation

**Date:** 2026-01-10
**Auditor:** Claude Code (Modo Martillo Nuclear)
**Branch:** `claude/audit-codex-state-pollution-0Q9yK`

## Executive Summary

Auditoría del problema de no-convergencia en el simulador de impacto ferroviario (Capítulo 8). El síntoma original: 2 convergidos de 101 casos, 99 fallidos.

### Hallazgos Principales

| Causa | Estado | Impacto |
|-------|--------|---------|
| Escalamiento patológico del residual en step 0 | **CORREGIDO** | Crítico para step 0 |
| Aislamiento de estado entre retries | **VERIFICADO OK** | N/A |
| Discontinuidad de fricción en v=0 | **IDENTIFICADO** | Causa dominante de fallos durante contacto |

## Parte A: Reproducción del Problema

### Micro Batch Inicial (ANTES del fix)

```
python scripts/run_parametric_study.py --section 8.3 --max-cases 8 --seed 42
```

**Resultado:**
- Converged: **1/8**
- Failed: **7/8**
- Fallos en step 0: **0** (todos fallan durante contacto)
- Fallos durante contacto: **7/8**

### Análisis de Patrones de Fallo

Todos los fallos ocurren:
- Durante contacto activo (in_contact: true)
- Cerca del punto de rebote (v_front ≈ 0)
- Con penetración alta (~270-350 mm)
- Con fuerza de contacto alta (~15-21 MN)

## Parte B: Verificación del Aislamiento de Estado

### initial_state.json - Comparación entre Attempts

| Campo | Attempt 0 | Attempt 1 | Attempt 2 | Verificación |
|-------|-----------|-----------|-----------|--------------|
| engine_id | 139569632907024 | 139569630448528 | 139569585536720 | ✓ Diferentes |
| integrator_id | 139569630067280 | 139569585533840 | 139569585533968 | ✓ Diferentes |
| q_id | 139569628179184 | 139569628183312 | 139569628180816 | ✓ Diferentes |
| qp_id | 139569628181872 | 139569628183792 | 139569628182544 | ✓ Diferentes |
| x_front | 12.0 | 12.0 | 12.0 | ✓ Idénticos |
| v_front | -22.222 | -22.222 | -22.222 | ✓ Idénticos |
| q_norm | 789.577 | 789.577 | 789.577 | ✓ Idénticos |
| qp_norm | 83.148 | 83.148 | 83.148 | ✓ Idénticos |

**Conclusión:** El aislamiento de estado entre attempts funciona correctamente. El fix de Codex para esto está OK.

## Parte C: Causas Identificadas

### C1. Escalamiento del Residual en Step 0 (CORREGIDO)

**Problema:** En `engine.py:1262`, la referencia del residual era:
```python
ref = float(np.linalg.norm(R_total_old) + 1.0)
```

Cuando el sistema empieza en equilibrio:
- `R_total_old = 0` (sin fuerzas internas/externas)
- Con stiffness damping: `C @ v0 = 0` (modo rígido)
- `ref = 0 + 1.0 = 1.0` (patológicamente pequeño)
- Cualquier residual pequeño produce error relativo enorme

**Fix implementado:**
```python
force_rhs = state0.get("force_rhs")
ref_rhs = float(np.linalg.norm(force_rhs)) if force_rhs is not None else 0.0
ref_R = float(np.linalg.norm(R_total_old))
ref = max(ref_R, ref_rhs, 1.0)
```

Esto usa el RHS de la ecuación de equilibrio actual como referencia alternativa.

**Archivo modificado:** `src/railway_simulator/core/engine.py`
- Línea ~1232: Agregado `"force_rhs": force` al state dict
- Línea ~1262-1270: Nueva lógica de escalamiento

### C2. Discontinuidad de Fricción (IDENTIFICADO, NO CORREGIDO)

**Problema:** Los modelos de fricción (LuGre, Coulomb-Stribeck) usan `sign(v)` que tiene una discontinuidad en v=0. Cuando la velocidad cruza cero durante el rebote:

1. La fuerza de fricción cambia de dirección abruptamente
2. El Jacobiano de Newton no puede capturar esta discontinuidad
3. Newton no converge a través del punto de stick-slip

**Evidencia:**
- Fallos consistentemente cerca de v_front ≈ 0
- Aumentar iteraciones o relajar tolerancia no resuelve completamente
- Sin fricción, casos similares convergen más lejos

**Solución propuesta (NO implementada):**
- Regularizar `sign(v)` con función suave: `v / sqrt(v² + ε²)`
- O usar tolerancia adaptiva que se relaje cerca de v=0

### C3. Newton Jacobian Mode (VERIFICADO)

`newton_jacobian_mode: "per_step"` vs `"each_iter"`:
- `each_iter` es más robusto durante contacto pero más lento
- No resuelve completamente el problema de fricción

## Parte D: Tests Implementados

Archivo: `tests/test_newton_residual_scaling.py`

| Test | Estado | Descripción |
|------|--------|-------------|
| test_newton_step0_converges_with_rayleigh_damping | ✓ PASS | Verifica convergencia en step 0 con Rayleigh damping |
| test_newton_residual_ref_not_pathologically_small | ✓ PASS | Verifica que max_residual < 1.0 |
| test_newton_linear_case_converges_quickly | ✓ PASS | Caso lineal converge en 1-3 iteraciones |

## Parte E: Resultados Cuantitativos

### Antes del Fix (Step 0 con stiffness damping)
- Error reportado: `err = 1.2` a `7000+`
- Causa: `ref = 1.0` cuando debería ser `~17.5 MN`

### Después del Fix (Step 0)
- Error reportado: `err = 1.78e-6`
- Pasa step 0 correctamente

### Micro Batch Después del Fix
- Converged: **1/8** (sin cambio)
- La causa dominante de fallos NO es step 0, sino fricción durante contacto

## Recomendaciones

### Inmediatas (Para mejorar tasa de convergencia)

1. **Usar Rayleigh damping** en lugar de stiffness damping:
   ```yaml
   damping_model: rayleigh  # en lugar de 'stiffness'
   ```

2. **Relajar tolerancia de Newton** para casos con fricción alta:
   ```yaml
   newton_tol: 1e-4  # en lugar de 1e-6
   ```

3. **Usar Newton directo** en lugar de Picard:
   ```yaml
   solver: newton  # en lugar de 'picard'
   ```

### Futuras (Para resolver problema de fricción)

1. Regularizar modelos de fricción cerca de v=0
2. Implementar tolerancia adaptiva durante contacto
3. Considerar métodos semi-smooth Newton para problemas de stick-slip

## Archivos Modificados

1. `src/railway_simulator/core/engine.py`:
   - Línea ~1232: `"force_rhs": force` en state dict
   - Líneas ~1262-1270: Nueva lógica de escalamiento del residual

2. `tests/test_newton_residual_scaling.py`: Nuevo archivo con 3 tests

## Conclusión

El fix implementado corrige el bug de escalamiento del residual en step 0, verificado con tests automatizados. Sin embargo, la causa dominante de la baja tasa de convergencia (1/8) es la discontinuidad de fricción en v=0, que requiere modificaciones más profundas en el modelo de fricción.

**Criterio de éxito:**
- ✓ Step 0 ya no falla por escalamiento patológico
- ✗ Tasa de convergencia sigue siendo 1/8 (causa: fricción, no step 0)

---
*Modo Martillo: Si no hay test, no hay fix.*
