/**
 * Experimental Compute particle runtime. Opt in through the particle-compute feature.
 * The existing ParticleSystem CPU and analytic GPU processors remain unchanged.
 * @packageDocumentation
 */
import ParticleSystemRendererCompute from '../cocos/particle/renderer/particle-system-renderer-compute';
import { registerParticleComputeRendererFactory } from '../cocos/particle/renderer/particle-system-renderer-data';

registerParticleComputeRendererFactory((info) => new ParticleSystemRendererCompute(info));

export { ParticleSimulationBackend } from '../cocos/particle/enum';
export { ComputeSimControlOffset, ParticleComputeSimulation } from '../cocos/particle/compute/particle-compute-simulation';
export { compileParticleComputeBirth } from '../cocos/particle/compute/particle-compute-birth';
export { compileParticleComputeModules } from '../cocos/particle/compute/particle-compute-modules';
export { ParticleComputeTrail, compileParticleComputeTrail } from '../cocos/particle/compute/particle-compute-trail';
export { default as ParticleComputeModel } from '../cocos/particle/models/particle-compute-model';
